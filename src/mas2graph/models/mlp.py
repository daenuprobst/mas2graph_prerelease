from typing import Callable, List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)


class MLP(Module):
    def __init__(self, n_input_channels, n_hidden_channels, n_out_channels):
        super().__init__()

        self.layers = Sequential(
            BatchNorm1d(n_input_channels),
            Linear(n_input_channels, n_hidden_channels),
            Dropout(0.5),
            ReLU(),
            Linear(n_hidden_channels, n_hidden_channels // 2),
            Dropout(0.5),
            ReLU(),
            Linear(n_hidden_channels // 2, n_hidden_channels // 2),
            Dropout(0.5),
            ReLU(),
            Linear(n_hidden_channels // 2, n_out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class LightningMLPRegressor(pl.LightningModule):
    def __init__(
        self,
        n_input_channels: int,
        n_hidden_channels: int,
        learning_rate: float = 0.001,
    ) -> None:
        super(LightningMLPRegressor, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.model = MLP(n_input_channels, n_hidden_channels, 1)

        # Metrics
        self.train_r2 = R2Score()
        self.train_pearson = PearsonCorrCoef()
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.val_pearson = PearsonCorrCoef()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        self.test_pearson = PearsonCorrCoef()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)
        out = torch.flatten(out)
        loss = F.mse_loss(out, y)

        # Metrics
        self.train_r2(out, y)
        self.train_pearson(out.to(self.device), y.to(self.device))
        self.train_rmse(out, y)
        self.train_mae(out, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True)
        self.log("train/pearson", self.train_pearson, on_step=False, on_epoch=True)
        self.log("train/rmse", self.train_rmse, on_step=False, on_epoch=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        out = self(x)
        out = torch.flatten(out)
        loss = F.mse_loss(out, y)

        # Metrics
        self.val_r2(out, y)
        self.val_pearson(out.to(self.device), y.to(self.device))
        self.val_rmse(out, y)
        self.val_mae(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True)
        self.log("val/pearson", self.val_pearson, on_step=False, on_epoch=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        out = self(x)
        out = torch.flatten(out)
        loss = F.mse_loss(out, y)

        # Metrics
        self.test_r2(out, y)
        self.test_pearson(out.to(self.device), y.to(self.device))
        self.test_rmse(out, y)
        self.test_mae(out, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True)
        self.log("test/pearson", self.test_pearson, on_step=False, on_epoch=True)
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    # return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val/rmse"}
