import pickle
from multiprocessing import cpu_count

import lightning.pytorch as pl
import typer
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader

from molsetrep.models import LightningSRRegressor

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(train_file: str, valid_file: str, test_file: str):
    # This loads the data sets that have been preprocessed with the preprocessing script
    train = pickle.load(open(train_file, "rb"))
    valid = pickle.load(open(valid_file, "rb"))
    test = pickle.load(open(test_file, "rb"))

    dataset_train = train["set_data"]
    dataset_valid = valid["set_data"]
    dataset_test = test["set_data"]

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=cpu_count() if cpu_count() < 8 else 8,
        drop_last=True,
    )

    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=64,
        shuffle=False,
        num_workers=cpu_count() if cpu_count() < 8 else 8,
        drop_last=True,
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=64,
        shuffle=False,
        num_workers=cpu_count() if cpu_count() < 8 else 8,
        drop_last=True,
    )

    if len(dataset_train[0][0].shape) > 1:
        d = [len(dataset_train[0][i][0]) for i in range(len(dataset_train[0]) - 1)]
    else:
        d = [dataset_train[0][0].shape[0]]

    model = LightningSRRegressor(
        [128], [8], d, set_layer="transformer", learning_rate=0.00001
    )

    wandb_logger = WandbLogger(project="mas2graph_dejonge_qed_set")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=150, logger=wandb_logger, callbacks=[checkpoint_callback], devices=1
    )

    trainer.fit(
        model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_valid
    )
    trainer.test(dataloaders=data_loader_test, ckpt_path="best")


if __name__ == "__main__":
    app()
