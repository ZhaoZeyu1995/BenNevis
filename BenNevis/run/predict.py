"""
This script is used to conduct prediction given a checkpoint, a dataset, a language directory and an output directory.

Usage:
    torchrun --standalone --nproc_per_node=<ngpu> predict.py <data_dir> <lang_dir> <ckpt_path> <output_dir> [--not_pin_memory] [--batch_size <batch_size>] [--num_workers <num_workers>]

Arguments:
    data_dir: str
        The directory of the dataset to predict on
    lang_dir: str
        The directory of the language directory
    ckpt_path: str
        The path to the checkpoint
    output_dir: str
        The directory to save the predictions.
    --not_pin_memory: bool
        Whether to pin memory when loading data. By default, we pin memory.
    --batch_size: int
        The batch size for the prediction dataloader, by default 4.
    --num_workers: int
        The number of workers for the prediction dataloader, by default 4.
Example:
    torchrun --standalone --nproc_per_node=4 predict.py /path/to/data /path/to/lang /path/to/ckpt /path/to/output --not_pin_memory --batch_size 4 --num_workers 4

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import os
import torch
import logging
import argparse
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from BenNevis.core.dataset import Dataset, CollateFunc
from BenNevis.core.trainer import Trainer
from BenNevis.utils.misc import dynamic_import


def main(args):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["RANK"]))
    gpu_id = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(gpu_id)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    conf = ckpt["CONFIG"]

    predict_ds = Dataset(
        args.data_dir,
        args.lang_dir,
        load_wav=conf.data.load_wav,
        load_feats=conf.data.load_feats,
    )
    collate_fn = CollateFunc(
        load_wav=predict_ds.load_wav,
        load_feats=predict_ds.load_feats,
        ctc_target=predict_ds.ctc_target,
    )
    predict_dl = torch.utils.data.DataLoader(
        predict_ds,
        pin_memory=args.pin_memory,
        shuffle=False,
        batch_size=args.batch_size,
        sampler=DistributedSampler(predict_ds, shuffle=False),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model_class = dynamic_import(conf.model.module, conf.model.name)
    model = model_class(
        **conf.model["kwargs"],
    )
    model.to(device)

    trainer = Trainer(
        model,
    )

    trainer.predict(
        predict_dl,
        output_dir=args.output_dir,
        checkpoint=args.ckpt_path,
    )

    if world_size > 1:
        dist.barrier()
        if gpu_id == 0:
            logging.info("Merging predictions from multiple GPUs")
            os.system(
                f"cat {args.output_dir}/split{world_size}/ref.wrd.*.trn | awk '!seen[$0]++' > {args.output_dir}/ref.wrd.trn"
            )
            os.system(
                f"cat {args.output_dir}/split{world_size}/output.*.scp | awk '!seen[$1]++' > {args.output_dir}/output.scp"
            )
            with open(f"{args.output_dir}/ref.wrd.trn") as file:
                num_ref = len(file.readlines())
            with open(f"{args.output_dir}/output.scp") as file:
                num_scp = len(file.readlines())
            assert (
                num_ref == num_scp
            ), f"Number of lines in {args.output_dir}/ref.wrd.trn ({num_ref}) and {args.output_dir}/output.scp ({num_scp}) do not match"
            logging.info(f"Finally got {num_ref} samples in ref.wrd.trn and output.scp")

    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="""
            Conduct prediction given a checkpoint, a dataset, a language directory and an output directory.
            Usage:
            torchrun --standalone --nproc_per_node=<ngpu> predict.py <data_dir> <lang_dir> <ckpt_path> <output_dir> [--not_pin_memory] [--batch_size <batch_size>] [--num_workers <num_workers>]

            Example:
            torchrun --standalone --nproc_per_node=4 predict.py /path/to/data /path/to/lang /path/to/ckpt /path/to/output --not_pin_memory --batch_size 4 --num_workers 4
            """
    )
    parser.add_argument(
        "data_dir", type=str, help="The directory of the dataset to predict on"
    )
    parser.add_argument(
        "lang_dir", type=str, help="The directory of the language directory"
    )
    parser.add_argument("ckpt_path", type=str, help="The path to the checkpoint")
    parser.add_argument(
        "output_dir", type=str, help="The directory to save the predictions."
    )
    parser.add_argument(
        "--not_pin_memory",
        action="store_false",
        dest="pin_memory",
        help="Whether to pin memory when loading data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size for the prediction dataloader",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of workers for the prediction dataloader",
    )
    args = parser.parse_args()

    main(args)
