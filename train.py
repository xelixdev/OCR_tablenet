"""Training Module."""

import albumentations as album
import torch
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import joblib
from tablenet import MarmotDataModule
from tablenet import TableNetModule

if __name__ == "__main__":
    image_size = (896, 896)
    transforms_augmentation = album.Compose(
        [
            album.Resize(1024, 1024, always_apply=True),
            album.RandomResizedCrop(*image_size, scale=(0.7, 1.0), ratio=(0.7, 1)),
            album.HorizontalFlip(),
            album.VerticalFlip(),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    transforms_preprocessing = album.Compose(
        [album.Resize(*image_size, always_apply=True), album.Normalize(), ToTensorV2()]
    )

    complaint_dataset = MarmotDataModule(
        data_dir="data/blend/xelix_data",
        transforms_preprocessing=transforms_preprocessing,
        transforms_augmentation=transforms_augmentation,
        batch_size=1,
        num_workers=8,
    )  # type: ignore

    model = TableNetModule(batch_norm=False)

    EXPERIMENT_NAME = f"{model.__class__.__name__}"
    logger = TensorBoardLogger("tb_logs", name=EXPERIMENT_NAME)

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=5, save_last=True, mode="min"
    )
    early_stop_callback = EarlyStopping(monitor="validation_loss", mode="min", patience=10)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
        logger=logger,
        max_epochs=200,
        gpus=1 if torch.cuda.is_available() else None,
        default_root_dir="checkpoints/"
    )
    trainer.fit(model, datamodule=complaint_dataset)
    trainer.test()
    trainer.save_checkpoint("model.ckpt")


