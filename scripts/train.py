import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import lightning as L
import torchvision.models.segmentation as models
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from smolseg.module import SegmentationModule
from smolseg.data.cityscapes import Cityscapes


def train(
    root_dir,    
    max_epochs,
    run_name,
    batch_size,
    num_workers,
):
    train_dataset = Cityscapes(root_dir=root_dir, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataset = Cityscapes(root_dir=root_dir, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = models.deeplabv3_mobilenet_v3_large(num_classes=Cityscapes.NUM_CLASSES)
    module = SegmentationModule(
        model=model,
        device="cuda",
        hparams={
            "lr": 3e-4,
            "weight_decay": 1e-3,
            "num_classes": Cityscapes.NUM_CLASSES,
            "max_epochs": max_epochs,
        },
        train_dataloader=train_dataloader,
    )

    # Training configuration.
    log_dir = "/workspace/logs"
    loggers = [
        WandbLogger(
            project="smolseg",
            name=run_name,
            save_dir=os.path.join(log_dir, run_name),
            log_model=True,
        )
    ]
    callbacks = [
        ModelCheckpoint(
            monitor="val/miou",
            mode="max",
            save_top_k=1,
            dirpath=os.path.join(log_dir, run_name, "checkpoints"),
            filename="best-epoch={epoch:02d}",
            save_last=True,
        ),
        RichModelSummary(),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=max_epochs,
        profiler="simple",
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )

    trainer.fit(module, train_dataloader, val_dataloader)


if __name__ == '__main__':
    train(
        root_dir="/data/cityscapes",
        max_epochs=200,
        run_name="deeplabv3_mobilenet_v3_large",
        batch_size=8,
        num_workers=16,
    )


