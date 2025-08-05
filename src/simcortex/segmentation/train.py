# src/simcortex/segmentation/train.py

import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
import hydra

from simcortex.segmentation.data  import SegDataset
from simcortex.segmentation.model import Unet
from simcortex.segmentation.loss  import compute_dice


def train_one(cfg):
    # 1) Setup logging and TensorBoard
    os.makedirs(cfg.outputs.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg.outputs.log_dir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )
    tb_writer = SummaryWriter(cfg.outputs.log_dir)
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # 2) Data loaders
    train_loader = DataLoader(
        SegDataset(cfg, "train"),
        batch_size=cfg.trainer.img_batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers
    )
    val_loader = DataLoader(
        SegDataset(cfg, "val"),
        batch_size=cfg.trainer.img_batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers
    )

    # 3) Model, optimizer
    device = cfg.trainer.device
    model = Unet(c_in=cfg.model.in_channels, c_out=cfg.model.out_channels).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.learning_rate)

    # 4) Training loop
    for epoch in range(1, cfg.trainer.num_epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        tb_writer.add_scalar("Loss/train", avg_train, epoch)
        logging.info(f"Epoch {epoch}: Train Loss = {avg_train:.4f}")

        # 5) Validation
        if epoch % cfg.trainer.validation_interval == 0:
            model.eval()
            val_losses, val_dices = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    lval = nn.CrossEntropyLoss()(logits, y).item()
                    val_losses.append(lval)

                    pred = logits.argmax(dim=1)
                    # one-hot, drop background channel
                    pred_oh = F.one_hot(pred, num_classes=cfg.model.out_channels) \
                                .permute(0, 4, 1, 2, 3)[:, 1:]
                    y_oh = F.one_hot(y, num_classes=cfg.model.out_channels) \
                                .permute(0, 4, 1, 2, 3)[:, 1:]
                    dice = compute_dice(pred_oh.float(), y_oh.float())
                    val_dices.append(dice)

            avg_val = np.mean(val_losses)
            avg_dice = np.mean(val_dices)
            tb_writer.add_scalar("Loss/val", avg_val, epoch)
            tb_writer.add_scalar("Dice/val", avg_dice, epoch)
            logging.info(f"Epoch {epoch}: Val Loss = {avg_val:.4f}, Dice = {avg_dice:.4f}")

    # 6) Save final model
    torch.save(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        cfg.outputs.model_path
    )
    logging.info("Saved model to %s", cfg.outputs.model_path)
    tb_writer.close()


@hydra.main( version_base="1.1", config_path="../../../configs/segmentation", config_name="train" )
def train_app(cfg):
    train_one(cfg)


if __name__ == "__main__":
    train_app()
