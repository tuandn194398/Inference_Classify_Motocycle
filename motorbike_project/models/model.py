import time
from openpyxl import Workbook

import motorbike_project as mp
import torch.nn as nn
import torch
import pytorch_lightning as pl
import pandas as pd
from torchmetrics import ConfusionMatrix, Precision

from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from torchmetrics.classification import MulticlassPrecision, MulticlassAUROC

# Tạo một workbook mới
workbook = Workbook()
# Chọn sheet hiện tại
sheet = workbook.active
# Ghi tiêu đề hàng
headers = ['acc', 'loss', 'f1', 'inf_time', 'precision', 'auroc']
sheet.append(headers)


class MotorBikeModel(pl.LightningModule):
    def __init__(self, labels_csv_path: str, model: str = 'resnet152', num_classes: int = 4, lr: float = 1e-4):
        super().__init__()

        if model == 'resnet50':
            self.model = mp.ResNet50(num_classes=num_classes)
        elif model == 'vit':
            self.model = mp.VisionTransformerBase(num_classes=num_classes)
        elif model == 'vit_tiny':
            self.model = mp.VisionTransformerTiny(num_classes=num_classes)
        elif model == 'swinv2_base':
            self.model = mp.SwinV2Base(num_classes=num_classes)
        elif model == 'mobilenetv3_large':
            self.model = mp.MobileNetV3Large(num_classes=num_classes)
        elif model == 'resnet18':
            self.model = mp.ResNet18(num_classes=num_classes)

        # TODO: Add more models here if you want
        self.model_name = model
        self.train_loss = mp.RunningMean()
        self.val_loss = mp.RunningMean()

        self.train_acc = mp.RunningMean()
        self.val_acc = mp.RunningMean()

        self.train_f1 = mp.RunningMean()
        self.val_f1 = mp.RunningMean()

        self.val_precision = mp.RunningMean()
        self.val_auroc = mp.RunningMean()

        self.loss = nn.CrossEntropyLoss(
            weight=self._create_class_weight(labels_csv_path=labels_csv_path)
        )
        self.lr = lr
        # self.class_names = ['1', '2', '3', '4']
        self.inf_time = []
        self.precision = MulticlassPrecision(num_classes=4)
        self.auroc =  MulticlassAUROC(num_classes=4, average="macro", thresholds=None)

    def _create_class_weight(self, labels_csv_path: str):
        """
            Create class weight for the loss function
        """
        df = pd.read_csv(labels_csv_path)
        df.loc[df['answer'] > 3, 'answer'] = 1
        class_weights = class_weight.compute_class_weight('balanced', classes=df['answer'].unique(), y=df['answer'])
        return torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, x):
        starttime = time.time()
        x = self.model(x)
        t = time.time() - starttime
        self.log("infer_time_on_batch", t, prog_bar=True, sync_dist=True)
        self.inf_time.append(t)

        return x

    def _cal_loss_and_acc(self, batch):
        """
            Calculate loss and accuracy for a batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        with torch.no_grad():  # No need to calculate gradient here
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            f1 = f1_score(y_true=y.cpu(), y_pred=y_hat.argmax(dim=1).cpu(), average='macro')

        return loss, acc, f1

    def _cal_precision_auroc(self, batch):
        x, y = batch
        y_hat = self(x)
        # auroc = self.auroc(y_hat, y)
        auroc = 0
        precision = self.precision(y_hat, y)
        return precision, auroc

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)
        self.train_loss.update(loss.item(), batch[0].shape[0])
        self.train_acc.update(acc.item(), batch[0].shape[0])
        self.train_f1.update(f1, batch[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)
        self.val_loss.update(loss.item(), batch[0].shape[0])
        self.val_acc.update(acc.item(), batch[0].shape[0])
        self.val_f1.update(f1, batch[0].shape[0])

        precision, auroc = self._cal_precision_auroc(batch)
        # self.val_auroc.update(auroc.item(), batch[0].shape[0])
        self.val_precision.update(precision.item(), batch[0].shape[0])
        # self.log("val_auroc_batch", self.val_auroc(), sync_dist=True)
        self.log("val_precision_batch", precision.item(), prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss(), sync_dist=True)
        self.log("train_acc", self.train_acc(), sync_dist=True)
        self.log("train_f1", self.train_f1(), sync_dist=True)
        self.train_loss.reset()
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("infer_time_on_epoch", sum(self.inf_time), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", self.val_loss(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_precision_epoch", self.val_precision(), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        # self.log("val_auroc_epoch", self.val_auroc(), sync_dist=True)

        # Ghi dữ liệu vào excel
        row = []
        row = [self.val_acc(), self.val_loss(), self.val_f1(), sum(self.inf_time), self.val_precision(), 123.01]
        print("Validation is:", row)
        #   'acc', 'loss', 'f1', 'inf_time', 'precision', 'auroc'
        # Đọc workbook
        # workbook2 = Workbook('data.xlsx')
        # sheet2 = workbook2.active
        sheet.append(row)
        workbook.save(f"{self.model_name}_13kbb4cls_100eps.csv")
        print("Hoàn thành ra excel!")

        self.inf_time = []
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        # self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
