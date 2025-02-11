from typing import Callable, List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from molsetrep.metrics import AUPRC
from torch.nn import (
    BatchNorm1d,
    CrossEntropyLoss,
    Dropout,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torchmetrics.classification import AUROC, Accuracy, F1Score
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)

from torch_geometric.nn.models import GAT
from torch_geometric.nn import GINEConv, MLP, MessagePassing, global_mean_pool
from torch_geometric.nn.models.basic_gnn import BasicGNN


class LightningGATRegressor(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        n_hidden_channels: int,
        num_layers: int,
        in_edge_channels,
        learning_rate: float = 0.001,
    ) -> None:
        super(LightningGATRegressor, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.model = GAT(
            in_channels,
            n_hidden_channels,
            num_layers,
            1,
            edge_dim=in_edge_channels,
            v2=True,
        )

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

    def forward(self, x, edge_index, edge_attr=None):
        return self.model(x, edge_index, edge_attr=edge_attr)

    def training_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
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

    def validation_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
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

    def test_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
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
