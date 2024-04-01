"""
This file contains the Trainer class which is responsible for training the model.
Basically, it is a wrapper around the model, optimizer and the data loader, etc.
It supports distributed training using PyTorch's DistributedDataParallel (DDP) module.
Even though the code is written to support distributed training, it can also be used for single GPU training
by simply setting --nproc_per_node=1 in the torchrun command.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import os
import torch
import math
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging
import wandb
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from torchinfo import summary
from kaldiio import WriteHelper
from BenNevis.core.losses import GraphLoss
from BenNevis.samplers.dynamic import DistributedSyncDynamicBatchSampler


class Trainer:
    """
    This class is used to train models.
    It supports distributed training using PyTorch's DistributedDataParallel (DDP) module.

    Arguments
    ---------
    model: torch.nn.Module
        The model to be trained.
    optimizers: Optional[List[torch.optim.Optimizer]]
        A list of optimizers to be used for training.
    schedulers: Optional[List[torch.optim.lr_scheduler._LRScheduler]]
        A list of learning rate schedulers to be used for training.
    loss_func: Optional[GraphLoss]
        The loss function to be used for training.
        Currently, this should be an instance of GraphLoss.
    exp_dir: Optional[str]
        The directory where the experiment logs and checkpoints will be saved.
    save_every_n_epochs: Optional[float]
        The model will be saved every n epochs.
        Example: if save_every_n_epochs=0.5, the model will be saved every 0.5 epochs.
    save_every_n_steps: Optional[int]
        The model will be saved every n steps.
        Note that this will be ignored if save_every_n_epochs is set.
    save_top_k: Optional[int]
        The top k models will be saved based on the validation loss.
    max_epochs: Optional[int]
        The maximum number of epochs to train the model.
    max_steps: Optional[int]
        The maximum number of steps to train the model.
    log_every_n_steps: Optional[int]
        The training logs will be logged every n steps.
    accum_grad_steps: Optional[int]
        The number of gradient accumulation steps, default is 1.
    grad_max_norm: Optional[float]
        The maximum norm of the gradients, default is None.
        torch.nn.utils.clip_grad_norm_ will be used if this is set.
    config: Optional[Dict[str, Any]]
        A dictionary containing the arguments for the whole training setting.
        This is useful for several aspects, such as logging, saving, and resuming the training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        schedulers: Optional[List[torch.optim.lr_scheduler._LRScheduler]] = None,
        loss_func: Optional[GraphLoss] = None,
        exp_dir: Optional[str] = None,
        save_every_n_epochs: Optional[float] = 1.0,
        save_every_n_steps: Optional[int] = None,
        save_top_k: Optional[int] = 5,
        max_epochs: Optional[int] = 100,
        max_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = 50,
        accum_grad_steps: Optional[int] = 1,
        grad_max_norm: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.gpu_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(self.gpu_id)

        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.loss_func = loss_func
        self.config = config

        if self.gpu_id == 0:
            summary(self.model, depth=6, verbose=1)

        self.exp_dir = exp_dir
        if self.exp_dir:
            os.makedirs(self.exp_dir, exist_ok=True)
        else:
            logging.warning(
                "No experiment directory is provided. Please make sure to only do prediction with the model."
            )

        if save_every_n_epochs:

            def check_valid(save_every_n_epochs):
                if isinstance(save_every_n_epochs, int):
                    if save_every_n_epochs >= 1:
                        return True
                    else:
                        return False
                elif isinstance(save_every_n_epochs, float):
                    if 0 < save_every_n_epochs and save_every_n_epochs < 1.0:
                        return True
                    elif (
                        save_every_n_epochs.is_integer() and save_every_n_epochs >= 1.0
                    ):
                        return True
                    else:
                        return False
                else:
                    return False

            if not check_valid(save_every_n_epochs):
                raise ValueError(
                    "save_every_n_epochs should be a positive float number smaller than 1.0. \
                        Otherwise it must be a positive integer."
                )

            self.save_every_n_epochs = save_every_n_epochs
            self.save_every_n_steps = None
            logging.info(
                f"RANK {self.gpu_id}: Model will be saved every {self.save_every_n_epochs} epochs"
            )
            if save_every_n_steps is not None:
                logging.warning(
                    f"RANK {self.gpu_id}: Both save_every_n_epochs and save_every_n_step are set. \
                            Ignoring save_every_n_step."
                )
        else:
            self.save_every_n_epochs = None
            self.save_every_n_steps = save_every_n_steps
            if save_every_n_steps:
                logging.info(
                    f"RANK {self.gpu_id}: Model will be saved every {self.save_every_n_steps} steps"
                )
            else:
                self.save_every_n_epochs = 1
                logging.warning(
                    f"RANK {self.gpu_id}: No save_every_n_epochs or save_every_n_steps is set. \
                        Model will be saved every 1 epoch by default."
                )

        if save_top_k:
            assert (
                isinstance(save_top_k, int) and save_top_k > 0
            ), "save_top_k should be a positive integer."
        else:
            save_top_k = None
        self.save_top_k = save_top_k

        self.max_epochs = max_epochs
        self.max_steps = max_steps

        self.log_every_n_steps = log_every_n_steps
        self.accum_grad_steps = accum_grad_steps
        self.grad_max_norm = grad_max_norm

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_best_checkpoint(self):
        """
        Load the best checkpoint based on the validation loss.
        """
        best_ckpt_path = os.path.join(self.ckpt, "best.ckpt")
        self._load_checkpoint(best_ckpt_path)

    def _load_checkpoint(self, ckpt_path: str):
        """
        Load the checkpoint from the given path.

        Arguments
        ---------
        ckpt_path: str
            The path to the checkpoint file.
        """
        logging.info(f"RANK {self.gpu_id}: Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["MODEL_STATE"])
        self.epoch = ckpt["EPOCH"]
        self.step = ckpt["STEP"]
        self.valid_loss = ckpt["VALID_LOSS"]
        self.config = ckpt["CONFIG"]
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(ckpt[f"OPTIMIZER_{i}"])
        for i, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(ckpt[f"SCHEDULER_{i}"])

    def _save_checkpoint(self, valid_loss: float):
        """
        Save the checkpoint.
        The checkpoint will be saved in the {self.ckpt_dir} directory.
        It has a name of the format:
        epoch_{epoch:d}_step_{step:d}_val_loss_{valid_loss:.4f}.pt
        Besides, keep a symbol link to the best checkpoint according to the validation loss.
        That said, the best checkpoint will be always accessible as {self.ckpt_dir}/best.pt.

        There are the following keys in the checkpoint:
            - CONFIG: the configuration dictionary.
            - MODEL_STATE: the state of the model.
            - EPOCH: the current epoch.
            - STEP: the current step.
            - VALID_LOSS: the current validation loss.
            - OPTIMIZER_{i}: the state of the i-th optimizer.
            - SCHEDULER_{i}: the state of the i-th scheduler.
        Note that the configuration is actually redundant, as it is also saved by WandB and Hydra.
        However, we would like to keep it here for the sake of completeness.
        It allows us to load a model with the checkpoint only, as it has the information for us to
        initiate a model and load the dict_state of the parameters.
        This is useful especially when we want to do prediction with the model.

        Arguments
        ---------
        valid_loss: float
            The current validation loss.
        """
        ckpt_name = (
            f"epoch_{self.epoch:d}_step_{self.step:d}_val_loss_{valid_loss:.4f}.pt"
        )
        best_ckpt = os.path.join(self.ckpt_dir, "best.pt")
        best_loss = self.top_k[0][0] if self.top_k else float("inf")
        last_ckpt = None
        if self.save_top_k:
            last_ckpt = self.top_k[-1] if len(self.top_k) == self.save_top_k else None
            self.top_k.append((valid_loss, ckpt_name))
            self.top_k = sorted(self.top_k, key=lambda x: x[0])[: self.save_top_k]
            if (valid_loss, ckpt_name) not in self.top_k:
                return
        snapshot = {
            "CONFIG": self.config,
            "MODEL_STATE": self.model.state_dict(),
            "EPOCH": self.epoch,
            "STEP": self.step,
            "VALID_LOSS": valid_loss,
        }
        for i, optimizer in enumerate(self.optimizers):
            snapshot[f"OPTIMIZER_{i}"] = optimizer.state_dict()
        for i, scheduler in enumerate(self.schedulers):
            snapshot[f"SCHEDULER_{i}"] = scheduler.state_dict()
        torch.save(snapshot, os.path.join(self.ckpt_dir, ckpt_name))
        logging.debug(
            f"RANK {self.gpu_id}: Saved checkpoint at {os.path.join(self.ckpt_dir, ckpt_name)}"
        )
        # delete the last checkpoint
        if last_ckpt is not None:
            os.remove(os.path.join(self.ckpt_dir, last_ckpt[1]))
        # update the best checkpoint
        if valid_loss < best_loss:
            if os.path.exists(best_ckpt):
                os.remove(best_ckpt)
            os.symlink(ckpt_name, best_ckpt)
            logging.debug(
                f"RANK {self.gpu_id}: Updated the best checkpoint at {best_ckpt}"
            )

    def _load_weights(self, ckpt_path: str):
        """
        Load the weights from the given checkpoint.

        Arguments
        ---------
        ckpt_path: str
            The path to the checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["MODEL_STATE"])

    def _compute_loss(
        self, batch: Dict[str, Any], pin_memory: bool = False
    ) -> torch.Tensor:
        """
        Compute the loss for the given batch.

        Arguments
        ---------
        batch: Dict[str, Any]
            The batch of data.
        pin_memory: bool
            Whether `batch` comes from a DataLoader with pin_memory=True.

        Returns
        -------
        loss: torch.Tensor
            The computed loss.
        """
        assert self.loss_func.reduction in [
            "mean",
            "sum",
        ], "Currently GraphLoss.reduction must \
                be either 'mean' or 'sum', but got {self.loss_func.reduction}"
        if "feats" in batch:
            inputs = batch["feats"].to(self.device, non_blocking=pin_memory)
            input_lens = batch["feats_lens"].to(self.device, non_blocking=pin_memory)
        elif "wavs" in batch:
            inputs = batch["wavs"].to(self.device, non_blocking=pin_memory)
            input_lens = batch["wav_lens"].to(self.device, non_blocking=pin_memory)
        else:
            raise ValueError(
                f"RANK {self.gpu_id}: Expected 'feats' or 'wavs' in batch, got {batch.keys()}"
            )

        outputs = self.model(inputs, input_lens)
        log_probs, log_prob_lens = outputs[0], outputs[1]

        target_lengths = batch["target_lengths"].to(
            self.device, non_blocking=pin_memory
        )
        loss = self.loss_func(
            log_probs,
            log_prob_lens,
            batch["word_ids"],
            target_lengths,
        )
        return loss

    def _train_batch(self, batch: Dict[str, Any], pin_memory: bool = False) -> float:
        """
        Train the model on the given batch.

        Arguments
        ---------
        batch: Dict[str, Any]
            The batch of data.
        pin_memory: bool
            Whether `batch` comes from a DataLoader with pin_memory=True.

        Returns
        -------
        loss_item: float
            The computed loss averaged on all GPUs.
        """
        self.model.train()
        loss = self._compute_loss(batch, pin_memory=pin_memory)
        loss.backward()
        if (self.step + 1) % self.accum_grad_steps == 0:
            if self.grad_max_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_max_norm
                )
            for opt in self.optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_item = loss.item() / self.world_size
        return loss_item

    def _valid_batch(self, batch: Dict[str, Any], pin_memory: bool = False) -> float:
        """
        Validate the model on the given batch.

        Arguments
        ---------
        batch: Dict[str, Any]
            The batch of data.
        pin_memory: bool
            Whether `batch` comes from a DataLoader with pin_memory=True.

        Returns
        -------
        loss_item: float
            The computed loss averaged on all GPUs.
        """
        self.model.eval()
        with torch.no_grad():
            loss = self._compute_loss(batch, pin_memory=pin_memory)

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_item = loss.item() / self.world_size
        return loss_item

    def _test_batch(self, batch: Dict[str, Any], pin_memory: bool = False) -> float:
        """
        Test the model on the given batch.

        Arguments
        ---------
        batch: Dict[str, Any]
            The batch of data.
        pin_memory: bool
            Whether `batch` comes from a DataLoader with pin_memory=True.

        Returns
        -------
        loss_item: float
            The computed loss averaged on all GPUs.
        """
        self.model.eval()
        with torch.no_grad():
            loss = self._compute_loss(batch, pin_memory)

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_item = loss.item() / self.world_size
        return loss_item

    def _predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the model on the given batch.

        Arguments
        ---------
        batch: Dict[str, Any]
            The batch of data.

        Returns
        -------
        prediction: Dict[str, Any]
            The prediction.
            Note that some keys in the dictionary are actually read directly from the batch.
            This is useful especially when we need to keep track of the original data.
            Here is the list of keys in the dictionary:
                - "log_probs": np.ndarray, of shape (batch_size, max_len, token_size)
                - "log_prob_lens": np.ndarray, of shape (batch_size,)
                - "names": List[str], of length batch_size, the utterance ids
                - "spks": List[str], of length batch_size, the speaker ids
                - "texts": List[str], of length batch_size, the original transcriptions
                - "batch_size": int, the batch size of the prediction,
                    which can be read from log_probs.shape[0] easily, but we keep it here for convenience
        """
        self.model.eval()
        if "feats" in batch:
            inputs = batch["feats"].to(self.device)
            input_lens = batch["feats_lens"].to(self.device)
        elif "wavs" in batch:
            inputs = batch["wavs"].to(self.device)
            input_lens = batch["wav_lens"].to(self.device)
        else:
            raise ValueError(
                f"RANK {self.gpu_id}: Expected 'feats' or 'wavs' in batch, got {batch.keys()}"
            )
        names = batch["names"]
        spks = batch["spks"]
        texts = batch["texts"]
        batch_size = batch["batch_size"]
        with torch.inference_mode():
            outputs = self.model(inputs, input_lens)
        log_probs, log_prob_lens = outputs[0], outputs[1]
        return {
            "log_probs": log_probs.cpu().detach().numpy(),
            "log_prob_lens": log_prob_lens.cpu().detach().numpy(),
            "names": names,
            "spks": spks,
            "texts": texts,
            "batch_size": batch_size,
        }

    def _train_epoch(self):
        """
        Train the model for one epoch.
        """
        # Set the epoch for the sampler or batch sampler
        self.train_sampler.set_epoch(self.epoch)

        num_steps = 0
        loss_value_sum = 0
        if self.gpu_id == 0:
            progress_bar = tqdm(total=self.num_samples, position=0, unit="samples")
            progress_bar.set_description(f"Epoch {self.epoch}/{self.max_epochs-1}")
            progress_bar.set_postfix(self.metrics_dict)
            progress_bar.refresh()

        epoch_num_batches = None
        for batch in self.train_dl:
            if epoch_num_batches is None:
                if isinstance(self.train_sampler, DistributedSampler):
                    epoch_num_batches = len(self.train_dl.batch_sampler)
                elif isinstance(self.train_sampler, DistributedSyncDynamicBatchSampler):
                    epoch_num_batches = self.train_sampler.num_batches
            batch_size = batch["batch_size"]
            loss_value = self._train_batch(batch, pin_memory=self.train_dl.pin_memory)
            num_steps += 1
            if self.gpu_id == 0:
                progress_bar.update(batch_size * self.world_size)
                self.metrics_dict["loss"] = loss_value
                progress_bar.set_postfix(self.metrics_dict)
                loss_value_sum += loss_value
                if self.step % self.log_every_n_steps == 0:
                    wandb.log(
                        {"loss_step": loss_value, "step": self.step}, step=self.step
                    )
                    for opt in self.optimizers:
                        wandb.log(
                            {
                                "lr_" + opt.nickname: opt.param_groups[0]["lr"],
                                "step": self.step,
                            },
                            step=self.step,
                        )

            # Save the model every n epochs, where n can be a float number
            if (
                num_steps % math.ceil(epoch_num_batches * self.save_every_n_epochs) == 0
                and num_steps != epoch_num_batches
            ):
                valid_loss = self._valid_epoch()
                if self.gpu_id == 0:
                    self._save_checkpoint(valid_loss)
                    wandb.log(
                        {"valid.loss_epoch": valid_loss, "step": self.step},
                        step=self.step,
                    )

            self.step += 1
            if self.max_steps:
                if self.step > self.max_steps:
                    logging.info(
                        f"RANK {self.gpu_id}: Reached maximum number of steps, exiting..."
                    )
                    self._exit()
            if self.save_every_n_steps:
                if self.step % self.save_every_n_steps == 0 and self.gpu_id == 0:
                    valid_loss = self._valid_epoch()
                    self._save_checkpoint(valid_loss)
                    wandb.log(
                        {"valid.loss_epoch": valid_loss, "step": self.step},
                        step=self.step,
                    )

        train_loss = loss_value_sum / num_steps
        if self.gpu_id == 0:
            self.metrics_dict["loss_epoch"] = train_loss
            progress_bar.set_postfix(self.metrics_dict)
            progress_bar.close()
        return train_loss

    def _valid_epoch(self):
        """
        Validate the model for one epoch.
        """
        num_steps = 0
        loss_value_sum = 0
        if self.gpu_id == 0:
            valid_progress_bar = tqdm(
                total=self.num_val_samples,
                desc="Validating",
                position=1,
                unit="samples",
            )

        for batch in self.valid_dl:
            batch_size = batch["batch_size"]
            loss_value = self._valid_batch(batch, pin_memory=self.valid_dl.pin_memory)
            num_steps += 1
            loss_value_sum += loss_value
            if self.gpu_id == 0:
                valid_progress_bar.update(batch_size * self.world_size)

        valid_loss = loss_value_sum / num_steps
        if self.gpu_id == 0:
            self.metrics_dict["val_loss"] = valid_loss
            valid_progress_bar.close()
        return valid_loss

    def _test_epoch(self):
        """
        Test the model for one epoch.
        """
        num_steps = 0
        loss_value_sum = 0
        if self.gpu_id == 0:
            self.progress_bar.total = self.num_test_samples

        for batch in self.test_dl:
            batch_size = batch["batch_size"]
            loss_value = self._test_batch(batch)
            num_steps += 1
            loss_value_sum += loss_value
            if self.gpu_id == 0:
                self.progress_bar.update(batch_size * self.world_size)

        test_loss = loss_value_sum / num_steps
        if self.gpu_id == 0:
            self.metrics_dict["test_loss"] = test_loss
            self.progress_bar.set_postfix(self.metrics_dict)
            self.progress_bar.reset()
        return test_loss

    def _predict_epoch(self, output_dir) -> None:
        """
        Predict the model for one epoch.

        Arguments
        ---------
        output_dir : str
            The directory to save the predictions.
        """
        if self.gpu_id == 0:
            progress_bar = tqdm(
                total=self.num_predict_samples,
                desc="Predicting",
                position=0,
                unit="samples",
            )

        if self.world_size > 1:
            predicted = set()
            with open(
                os.path.join(output_dir, "ref.wrd.%d.trn" % (self.gpu_id + 1)), "w"
            ) as y:
                yc = ""
                with WriteHelper(
                    "ark,scp:%s,%s"
                    % (
                        os.path.join(
                            os.getcwd(), output_dir, "output.%d.ark" % (self.gpu_id + 1)
                        ),
                        os.path.join(
                            os.getcwd(), output_dir, "output.%d.scp" % (self.gpu_id + 1)
                        ),
                    )
                ) as writer:
                    for batch in self.predict_dl:
                        prediction = self._predict_batch(batch)
                        if self.gpu_id == 0:
                            progress_bar.update(batch["batch_size"] * self.world_size)
                        log_probs = prediction["log_probs"]
                        log_prob_lens = prediction["log_prob_lens"]
                        names = prediction["names"]
                        spks = prediction["spks"]
                        texts = prediction["texts"]
                        batch_size = prediction["batch_size"]
                        for i in range(batch_size):
                            if names[i] in predicted:
                                continue
                            predicted.add(names[i])
                            log_prob = log_probs[i, : log_prob_lens[i], :]
                            writer(names[i], log_prob)
                            yc += "%s (%s-%s)\n" % (texts[i], spks[i], names[i])

                y.write(yc)
            logging.info(f"RANK {self.gpu_id}: Predicted {len(predicted)} samples")
        else:
            with open(os.path.join(output_dir, "ref.wrd.trn"), "w") as y:
                yc = ""
                with WriteHelper(
                    "ark,scp:%s,%s"
                    % (
                        os.path.join(os.getcwd(), output_dir, "output.ark"),
                        os.path.join(os.getcwd(), output_dir, "output.scp"),
                    )
                ) as writer:
                    for batch in self.predict_dl:
                        prediction = self._predict_batch(batch)
                        if self.gpu_id == 0:
                            progress_bar.update(batch["batch_size"])
                        log_probs = prediction["log_probs"]
                        log_prob_lens = prediction["log_prob_lens"]
                        names = prediction["names"]
                        spks = prediction["spks"]
                        texts = prediction["texts"]
                        batch_size = prediction["batch_size"]
                        for i in range(batch_size):
                            log_prob = log_probs[i, : log_prob_lens[i], :]
                            writer(names[i], log_prob)
                            yc += "%s (%s-%s)\n" % (texts[i], spks[i], names[i])

                y.write(yc)
            logging.info(f"Predicted {self.num_predict_samples} samples")

    def fit(
        self,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        checkpoint: str = None,
        load_weights_only: bool = False,
    ):
        """
        Train the model.

        Arguments
        ---------
        train_dl : DataLoader
            The training DataLoader.
        valid_dl : DataLoader
            The validation DataLoader.
        checkpoint : str
            The path to the checkpoint to load.
        load_weights_only : bool
            Whether to load only the weights of the model.
        """
        self.train_dl = train_dl
        self.valid_dl = valid_dl

        if isinstance(
            self.train_dl.batch_sampler, (DistributedSyncDynamicBatchSampler)
        ):
            self.num_samples = len(self.train_dl.batch_sampler) * self.world_size
            self.train_sampler = self.train_dl.batch_sampler
        elif isinstance(self.train_dl.sampler, DistributedSampler):
            self.num_samples = len(self.train_dl.sampler) * self.world_size
            self.train_sampler = self.train_dl.sampler
        else:
            raise ValueError(
                f"RANK {self.gpu_id}: Unsupported sampler for training. \
                Please make sure that the sampler is in [torch.utils.data.distributed.DistributedSampler, \
                BenNevis.samplers.dynamic.DistributedSyncDynamicBatchSampler]"
            )

        if isinstance(
            self.valid_dl.batch_sampler, (DistributedSyncDynamicBatchSampler)
        ):
            self.num_val_samples = len(self.valid_dl.batch_sampler) * self.world_size
        elif isinstance(self.valid_dl.sampler, DistributedSampler):
            self.num_val_samples = len(self.valid_dl.sampler) * self.world_size
        else:
            raise ValueError(
                f"RANK {self.gpu_id}: Unsupported sampler for validation. \
                Please make sure that the sampler is in [torch.utils.data.distributed.DistributedSampler, \
                BenNevis.samplers.dynamic.DistributedSyncDynamicBatchSampler]"
            )

        self.top_k = []
        self.best_valid_loss = 1e10
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._check_validation_sanity()

        self.epoch = 0
        self.step = 0

        if self.gpu_id == 0:
            self.wandb_dir = os.path.join(self.exp_dir)
            os.makedirs(self.wandb_dir, exist_ok=True)
            wandb.init(
                dir=self.wandb_dir,
                project="BenNevis-%s-%s"
                % (
                    os.path.basename(os.path.dirname(os.getcwd())),
                    os.path.basename(os.getcwd()),
                )
                if self.config.logger["project"] is None
                else self.config.logger["project"],
                name=self.config.logger["name"],
                config=dict(self.config),
            )
            self.metrics_dict = {}

        dist.barrier()

        if checkpoint:
            if load_weights_only:
                self._load_weights(checkpoint)
            else:
                self._load_checkpoint(checkpoint)

        while self.epoch < self.max_epochs:
            self.model.train()
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)
            train_loss = self._train_epoch()

            self.epoch += 1

            self.model.eval()
            with torch.no_grad():
                valid_loss = self._valid_epoch()
            for lrs in self.schedulers:
                lrs.step(valid_loss)

            if self.gpu_id == 0:
                wandb.log(
                    {
                        "train.loss_epoch": train_loss,
                        "valid.loss_epoch": valid_loss,
                        "epoch": self.epoch,
                        "step": self.step,
                    },
                    step=self.step,
                )

                if (
                    self.save_every_n_epochs > 1
                    and self.epoch % self.save_every_n_epochs == 0
                ):
                    self._save_checkpoint(valid_loss)
                else:
                    self._save_checkpoint(valid_loss)
        logging.info(f"RANK {self.gpu_id}: Training finished")
        if self.gpu_id == 0:
            wandb.finish()

    def predict(
        self,
        predict_dl: DataLoader,
        output_dir: str = None,
        checkpoint: str = None,
    ) -> None:
        """
        Perform prediction on the given DataLoader on each device (GPU).
        There are three files which will be saved in the output_dir:
            - ref.wrd.%d.trn: The reference transcriptions with the pattern "text (speaker-name)" for each sample, produced by device %d-1.
            - output.%d.scp: The scp file of the log probabilities for each sample predicted by device %d-1.
            - output.%d.ark: The log probabilities for each sample predicted by device %d-1 in Kaldi's ark format.
        Note that the deivce ids are 0-based but the scp and ark files are 1-based.

        Arguments
        ---------
        predict_dl: DataLoader
            The DataLoader to predict on.
        output_dir: str
            The directory to save the predictions.
        checkpoint: str
            The path to the checkpoint to load.
        """
        self.predict_dl = predict_dl
        assert isinstance(
            self.predict_dl.sampler, DistributedSampler
        ), f"RANK {self.gpu_id}: Unsupported sampler for prediction. \
            Please make sure that the sampler is torch.utils.data.distributed.DistributedSampler."
        self.num_predict_samples = len(self.predict_dl.sampler) * self.world_size
        if checkpoint:
            logging.info(f"RANK {self.gpu_id}: Loading checkpoint from {checkpoint}")
            self._load_weights(checkpoint)
        else:
            logging.warning(
                f"RANK {self.gpu_id}: No checkpoint is provided. \
                        Using the current model. Please make sure this is intended."
            )
        if self.world_size > 1:
            output_dir = os.path.join(output_dir, "split%d" % self.world_size)
        os.makedirs(output_dir, exist_ok=True)
        self._predict_epoch(output_dir)

    def _check_validation_sanity(self):
        """
        Check the validation set for sanity.
        This is to make sure that everything works as expected.
        """
        check_steps = 2
        if self.gpu_id == 0:
            progress_bar = tqdm(
                total=check_steps,
                unit="steps",
                position=1,
            )
            progress_bar.set_description("Sanity check on validation set")
        for batch in self.valid_dl:
            _ = self._valid_batch(batch, pin_memory=self.valid_dl.pin_memory)
            check_steps -= 1
            if self.gpu_id == 0:
                progress_bar.update(1)
            if check_steps <= 0:
                if self.gpu_id == 0:
                    progress_bar.close()
                return

    def _exit(self):
        """
        Clean up the distributed environment.
        """
        if self.gpu_id == 0:
            self.progress_bar.close()
        exit()
