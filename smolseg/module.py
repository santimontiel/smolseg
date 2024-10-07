import lightning as L
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from transformers import get_cosine_schedule_with_warmup


class SegmentationModule(L.LightningModule):

    def __init__(self, model, device, train_dataloader, hparams):
        super().__init__()
        self.model = model
        self.num_classes = 19
        self.train_loader = train_dataloader
        self.params = hparams

        # METRICS.
        self.train_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=255
        )
        self.train_accuracy = MulticlassAccuracy(
            num_classes=self.num_classes,
            ignore_index=255
        )
        self.val_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=255
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=self.num_classes,
            ignore_index=255
        )

        # LOSSES.
        self.losses = {
            "ce": nn.CrossEntropyLoss(ignore_index=255),
            "dice": smp.losses.DiceLoss(mode="multiclass", ignore_index=255),
        }


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        logits = self.model(images)
        predictions = torch.argmax(logits, dim=1)

        ce_loss = self.losses["ce"](logits, masks)
        dice_loss = self.losses["dice"](logits, masks)
        loss = ce_loss + dice_loss

        self.train_miou(predictions, masks)
        self.train_accuracy(predictions, masks)

        self.log("train/loss", loss)
        self.log("train/ce_loss", ce_loss)
        self.log("train/dice_loss", dice_loss)
        self.log("train/miou", self.train_miou, on_step=False, on_epoch=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        logits = self.model(images)
        predictions = torch.argmax(logits, dim=1)

        ce_loss = self.losses["ce"](logits, masks)
        dice_loss = self.losses["dice"](logits, masks)
        loss = ce_loss + dice_loss

        self.val_miou(predictions, masks)
        self.val_accuracy(predictions, masks)

        self.log("val/loss", loss)
        self.log("val/ce_loss", ce_loss)
        self.log("val/dice_loss", dice_loss)
        self.log("val/miou", self.val_miou, on_step=False, on_epoch=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True)

        return loss


    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"]
        )
        scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=len(self.train_loader),
            num_training_steps=self.params["max_epochs"] * len(self.train_loader)
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    