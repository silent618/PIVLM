import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from datasets.datamodule import CLIP_DataModule, PIVLM_DataModule
from datasets.datasets import CLIPDataset, CLIP_SSL_Dataset
from datasets.datasets import clip_ssl_collate_fn, clip_collate_fn
from datasets.transforms import DataTransforms, SSLDataTransforms
from backbones.encoder import BertEncoder, ImageEncoder
from PIVLM.PIVLM import PIVLM
from torch import distributed as dist
from datasets.constants import *
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = PIVLM.add_model_specific_args(parser)
    args = parser.parse_args()

    args.gpus = 1
    args.strategy = None # 'ddp'
    args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)

    datamodule = PIVLM_DataModule(CLIP_SSL_Dataset, DataTransforms,
                                  SSLDataTransforms, clip_ssl_collate_fn,
                                  args.data_pct, args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = PIVLM(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"output/ckpts/PIVLM/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"output")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="PIVLM", save_dir=logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)
