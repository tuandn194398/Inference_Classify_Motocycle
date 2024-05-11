import os
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import datetime
import motorbike_project as mp
import torch
import wandb
import pytorch_lightning as pl
import argparse

from torch.utils.flop_counter import FlopCounterMode

torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50',
                    help='model name')
parser.add_argument('--name', type=str, default=None,
                    help='name of the experiment')
parser.add_argument('--max_epochs', '-me', type=int, default=20,
                    help='max epoch')
parser.add_argument('--folder_path', '-fp', type=str, default='',
                    help='folder path')
parser.add_argument('--csv_path', '-csv', type=str, default='',
                    help='The path to the csv file containing the labels')
parser.add_argument('--batch_size', '-bs', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', '-l', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num_workers', '-nw', type=int, default=0,
                    help='number of workers')
parser.add_argument('--seed', '-s', type=int, default=42,
                    help='seed')
parser.add_argument('--wandb', '-w', default=False, action='store_true',
                    help='use wandb or not')
parser.add_argument('--wandb_key', '-wk', type=str,
                    help='wandb API key')


args = parser.parse_args()


def train(args):
    model_name = args.model
    pl.seed_everything(args.seed, workers=True)

    # Wandb (Optional)
    if args.wandb:
        now_time = datetime.datetime.now()
        now_time = now_time.strftime("%H:%M:%d%m")
        wandb.login(key=args.wandb_key)
        name = f"{model_name}-{args.max_epochs}-{args.batch_size}-{args.lr}-{now_time}"
        logger = WandbLogger(
            project='BKAI-motorbike',
            name=name,
            log_model='all'  # Log model checkpoint at the end of training
        )

    else:
        logger = None

    print(f'Current working directory: {os.getcwd()}')

    # Dataset
    train_dataset = mp.MotorBikeDataset(
        config_path='motorbike_project/config',
        session='train',
        labels_csv_path=args.csv_path,
        folder_path=args.folder_path
    )

    val_dataset = mp.MotorBikeDataset(
        config_path='motorbike_project/config',
        session='val',
        labels_csv_path=args.csv_path,
        folder_path=args.folder_path
    )

    # DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # inp = torch.randn(1, 3, 224, 224, device='cpu')
    # Model
    model = mp.MotorBikeModel(
        model=model_name,
        labels_csv_path=args.csv_path,
        num_classes=4,
        lr=args.lr
    )

    # flop_counter = FlopCounterMode(model, depth=1)
    # with flop_counter:
    #     model(inp).sum().backward()

    # Callbacks
    root_path = os.path.join('checkpoints', model_name)
    model_folder = args.name or args.model
    ckpt_path = os.path.join(root_path, f"{model_folder}/")

    os.makedirs(ckpt_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=ckpt_path,
        filename=f'{model_name}',
        save_top_k=3,
        mode='min'
    )  # Save top 3 best models with lowest val_loss

    lr_callback = LearningRateMonitor(logging_interval='step')


    # Trainer
    trainer = pl.Trainer(
        default_root_dir=root_path,
        logger=logger,                  # Wandb logger
        callbacks=[checkpoint_callback, lr_callback],
        gradient_clip_val=0.5,          # Gradient clipping
        max_epochs=args.max_epochs,     # Max epochs
        enable_progress_bar=True,       # Enable progress bar
        deterministic=True,             # Reproducibility
        log_every_n_steps=1,            # Log every 1 step
        precision='bf16-mixed',                   # Use mixed precision
    )

    torch.set_deterministic_debug_mode(True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Fit model
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()


if __name__ == '__main__':
    train(args)
