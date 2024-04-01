"""
Note that this script is supposed to be run with `torchrun`.
You may learn more about `torchrun` from the official documentation of PyTorch.
This is a template for training a model using BenNevis.
The script is designed to be run with Hydra, a powerful configuration management tool.
The script is divided into several sections, each of which is described below.

1. Importing libraries and setting up the environment
2. Defining the configuration
3. Creating the model
4. Creating the optimizer and learning rate scheduler
5. Creating the trainer
6. Loading the dataset
7. Training the model

Usage:
    torchrun --standalone --nproc_per_node=<ngpu> train.py [options]

Example:
    torchrun --standalone --nproc_per_node=4 train.py \
            data.lang=data/lang_topo hydra.run.dir=exp/model-topo logger.name=model-topo

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import torch
import logging
import os
import hydra
from typing import Tuple, List
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from BenNevis.core.lang import Lang
from BenNevis.core.dataset import Dataset, CollateFunc
from BenNevis.samplers.dynamic import DistributedSyncDynamicBatchSampler
from BenNevis.core.trainer import Trainer
from BenNevis.utils.random import setup_seed
from BenNevis.utils.misc import dynamic_import


def get_opt(
        opt_conf: dict,
        model: torch.nn.Module,
        ) -> Tuple[torch.optim.Optimizer, List[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Get and configure an optimizer and the corresponding lr_schedulers,
    given an optimiser configuration and a model.

    Arguments
    ---------
    opt_conf : dict
        The configuration for the optimizer.
        There is one key in opt_conf, "param", which is the name of the parameter to optimize.
        Please make sure the attribute "param" does exist in the model so that it can be retrieved
        by `getattr(model, opt_conf["param"], None)`.
    model : torch.nn.Module
        The model to optimize.

    Returns
    -------
    opt : torch.optim.Optimizer
        The optimizer.
    lr_schedulers : List[torch.optim.lr_scheduler.LRScheduler]
        The learning rate schedulers bound to the optimizer.
        It can be a list of multiple learning rate schedulers.
        Also, it can be an empty list if no learning rate scheduler is bound to the optimizer.
    """
    opt_class = dynamic_import(
        opt_conf["module"],
        opt_conf["name"],
    )
    if opt_conf["param"]:
        param = getattr(model, opt_conf["param"], None)
        assert param is not None, f"Model has no attribute {opt_conf['param']}"
        opt = opt_class(
            param.parameters(),
            **opt_conf["kwargs"],
        )
    else:
        opt = opt_class(
            model.parameters(),
            **opt_conf["kwargs"],
        )
    setattr(opt, "nickname", opt_conf["nickname"])
    lr_schedulers = []
    if "lr_schedulers" in opt_conf:
        for lrs_conf in opt_conf["lr_schedulers"]:
            lrs = get_lr_schedulers(lrs_conf, opt)
            lr_schedulers.append(lrs)
    return opt, lr_schedulers


def get_lr_schedulers(
        lrs_conf: dict,
        opt: torch.optim.Optimizer,
        ) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get and configure a learning rate scheduler, given a learning rate scheduler configuration and an optimizer.

    Arguments
    ---------
    lrs_conf : dict
        The configuration for the learning rate scheduler.
    opt : torch.optim.Optimizer
        The optimizer.

    Returns
    -------
    lrs : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler.
    """
    lrs_class = dynamic_import(
        lrs_conf["module"],
        lrs_conf["name"],
    )
    lrs = lrs_class(opt, **lrs_conf["kwargs"])
    return lrs


def get_dl(data_conf: dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get the training and validation dataloaders.

    Arguments
    ---------
    data_conf : dict
        The configuration for the dataset.

    Returns
    -------
    train_dl : torch.utils.data.DataLoader
        The training dataloader.
    valid_dl : torch.utils.data.DataLoader
        The validation dataloader.
    """
    train_ds = Dataset(data_conf["train_ds"],
                       data_conf["lang"],
                       ratio_th=data_conf["ratio_th"],
                       load_wav=data_conf["load_wav"],
                       load_feats=data_conf["load_feats"],
                       sort=data_conf["sort"],
                       min_duration=data_conf["min_dur"],
                       max_duration=data_conf["max_dur"],
                       )
    valid_ds = Dataset(data_conf["valid_ds"],
                       data_conf["lang"],
                       ratio_th=data_conf["ratio_th"],
                       load_wav=data_conf["load_wav"],
                       load_feats=data_conf["load_feats"],
                       min_duration=data_conf["min_dur"],
                       max_duration=data_conf["max_dur"],
                       )
    collate_fn = CollateFunc(
        load_wav=train_ds.load_wav,
        load_feats=train_ds.load_feats,
        ctc_target=train_ds.ctc_target,
    )
    if "train_batch_size" not in data_conf:
        logging.info("Using dynamic batch size as 'train_batch_size' is not specified.")
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            pin_memory=data_conf["pin_memory"],
            shuffle=False,
            batch_sampler=DistributedSyncDynamicBatchSampler(
                train_ds,
                shuffle=True,
                max_sum_dur=data_conf["max_sum_dur"],
            ),
            collate_fn=collate_fn,
            num_workers=data_conf["num_workers"],
        )
    else:
        logging.info(f"Using fixed batch size {data_conf['train_batch_size']} with DistributedSampler.")
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            pin_memory=data_conf["pin_memory"],
            shuffle=False,
            batch_size=data_conf["train_batch_size"],
            sampler=DistributedSampler(train_ds, shuffle=getattr(data_conf, "shuffle", True)),
            collate_fn=collate_fn,
            num_workers=data_conf["num_workers"],
        )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        pin_memory=data_conf["pin_memory"],
        shuffle=False,
        batch_size=data_conf["val_batch_size"],
        sampler=DistributedSampler(valid_ds, shuffle=False),
        collate_fn=collate_fn,
        num_workers=data_conf["num_workers"],
    )
    return train_dl, valid_dl


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "conf"),
    config_name="config",
)
def main(cfg):
    setup_seed(1368)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["RANK"]))
    gpu_id = dist.get_rank()
    device = torch.device(gpu_id)

    lang = Lang(cfg.data["lang"], load_topo=True, load_lexicon=True)

    loss_class = dynamic_import(
        cfg.loss["module"],
        cfg.loss["name"],
    )
    loss = loss_class(
        lang,
        **cfg.loss["kwargs"],
    )

    model_class = dynamic_import(
        cfg.model["module"],
        cfg.model["name"],
    )
    cfg.model["kwargs"]["odim"] = lang.num_nn_output
    cfg.data["load_wav"] = True if cfg.model["name"] in ["Wav2Vec2Model"] else False
    cfg.data["load_feats"] = True if cfg.model["name"] not in ["Wav2Vec2Model"] else False
    model = model_class(
        **cfg.model["kwargs"],
    )
    model.to(device)

    if getattr(cfg, "init_all", False):
        logging.info(f"Initialising all parameters in the model as got cfg.init_all {cfg.init_all}.")
        for name, param in model.named_parameters():
            if len(param.shape) > 1:
                torch.nn.init.kaiming_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

    optimisers = []
    lr_schedulers = []
    for opt_conf in cfg.opts:
        opt, lrs = get_opt(opt_conf, model)
        optimisers.append(opt)
        lr_schedulers.extend(lrs)

    train_dl, valid_dl = get_dl(cfg.data)

    trainer = Trainer(
        model=model,
        optimizers=optimisers,
        schedulers=lr_schedulers,
        loss_func=loss,
        **cfg.trainer,
        config=cfg,
    )

    trainer.fit(
        train_dl,
        valid_dl,
        checkpoint=getattr(cfg, "ckpt", None),
        load_weights_only=getattr(cfg, "load_weights_only", False),
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    main()
