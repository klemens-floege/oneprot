from typing import Dict
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint
from pytorch_lightning import LightningDataModule
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import product
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import os
import sys

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory and its parent to the Python path
sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from utils.downstream import save_results_to_csv, load_data, count_f1_max


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


class EmbeddingDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data = {}

    def setup(self, stage=None):
        all_inputs = load_data(self.cfg)
        for partition in self.cfg.evaluate_on:
            self.data[partition] = EmbeddingDataset(
                all_inputs[f"{partition}_emb"], all_inputs[f"{partition}_target"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.data["train"], batch_size=self.cfg.model.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.data["valid"], batch_size=self.cfg.model.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data["test"], batch_size=self.cfg.model.batch_size)


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EvaluationModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        if cfg.model_type == "esm2":
            input_dim = 1280
        elif cfg.model_type in ["oneprot_1","oneprot_2","oneprot_3","oneprot_4","oneprot_5","oneprot_6","oneprot_7"]:
            input_dim = 128
        else:
            input_dim = 256
        if self.cfg.task_name == "HumanPPI":
            input_dim = input_dim * 2

        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI"]:
            output_dim = 1
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
            if self.cfg.task_name == "EC":
                output_dim = 585
            elif self.cfg.task_name == "GO-BP":
                output_dim = 1943
            elif self.cfg.task_name == "GO-MF":
                output_dim = 489
            elif self.cfg.task_name == "GO-CC":
                output_dim = 320

            self.loss_fn = F.binary_cross_entropy_with_logits
        elif self.cfg.task_name in ["ThermoStability"]:
            output_dim = 1
            self.loss_fn = F.mse_loss
        else:  # multi_class
            if self.cfg.task_name in ["TopEnzyme"]:
                output_dim = 826
            elif self.cfg.task_name in ["DeepLoc10"]:
                output_dim = 10
            self.loss_fn = F.cross_entropy

        self.model = MLPHead(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "ThermoStability", "HumanPPI"]:
            y_hat = y_hat.squeeze(1)
            y = y.float()
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
            y_hat = y_hat.float()
            y = y.float()

        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "ThermoStability", "HumanPPI"]:
            y_hat = y_hat.squeeze(1)
            y = y.float()
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
            y_hat = y_hat.float()
            y = y.float()
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.callback_metrics["val_loss"]
        self.log("val_loss_epoch", avg_val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.model.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)

        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPI"]:
            y_hat = y_hat.squeeze(1)
            preds = torch.sigmoid(y_hat)
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
            preds = torch.sigmoid(y_hat)
        elif self.cfg.task_name in ["ThermoStability"]:
            preds = y_hat.squeeze(
                1
            )  # For regression, we don't need to apply any activation
        else:  # multi_class
            preds = torch.softmax(y_hat, dim=1)
            preds = torch.argmax(preds, dim=1)

        return preds, y


def evaluate(cfg: DictConfig, data_module: EmbeddingDataModule) -> Dict:
    model = EvaluationModule(cfg)

    # Determine the accelerator and devices based on GPU availability
    if torch.cuda.is_available():
        accelerator = "gpu"
        fit_devices = 1
        predict_devices = 1
    else:
        accelerator = "cpu"
        fit_devices = 1
        predict_devices = 1

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.model.early_stopping_patience,
        mode="min",
        verbose=True,
        log_rank_zero_only=True,
    )

    # Trainer for fitting (multi-GPU if available, otherwise CPU)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-checkpoint",
    )

    fit_trainer = pl.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator=accelerator,
        devices=fit_devices,
        strategy="auto",
        callbacks=[early_stop_callback, RichProgressBar(), checkpoint_callback],
    )

    fit_trainer.fit(model, data_module)

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    model = EvaluationModule.load_from_checkpoint(best_model_path)
    # Trainer for prediction (single GPU if available, otherwise CPU)
    predict_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=predict_devices,
        strategy="auto",
    )

    results = {}
    for partition in ["valid", "test"]:
        if partition == "valid":
            predictions = predict_trainer.predict(model, data_module.val_dataloader())
        else:
            predictions = predict_trainer.predict(
                model, getattr(data_module, f"{partition}_dataloader")()
            )

        if cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI"]:
            y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()
            y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()
            print(y_true.shape)
            accuracy = accuracy_score(y_true, y_pred > 0.5)
            f1_micro = f1_score(y_true, y_pred > 0.5, average="micro")
            auc = roc_auc_score(y_true, y_pred)
            results[f"{partition}_accuracy"] = accuracy

        elif cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
            y_pred = torch.cat([p[0] for p in predictions]).cpu()
            y_true = torch.cat([p[1] for p in predictions]).cpu()
            f1_max = count_f1_max(y_pred, y_true)
            results[f"{partition}_f1_max"] = f1_max

        elif cfg.task_name in ["ThermoStability"]:
            y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()
            y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            spearman_rho, _ = spearmanr(y_true, y_pred)
            results[f"{partition}_spearman_rho"] = spearman_rho

        else:  # multi_class
            y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()
            y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average="micro")

            results[f"{partition}_accuracy"] = accuracy
            results[f"{partition}_f1_micro"] = f1_micro

    return results


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="saprot_mlp.yaml",
)
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("CUDA is not available. Running on CPU.")

    # Generate all combinations of hyperparameters
    param_combinations = product(
        cfg.sweep.learning_rate,
        cfg.sweep.batch_size,
        cfg.sweep.max_epochs,
        cfg.sweep.task_name,
        cfg.sweep.model_type,
    )

    for (
        lr,
        batch_size,
        max_epochs,
        task_name,
        model_type,
    ) in param_combinations:
        # Update the configuration with the current hyperparameters
        cfg.model.learning_rate = lr
        cfg.model.batch_size = batch_size
        cfg.model.max_epochs = max_epochs
        cfg.task_name = task_name
        cfg.model_type = model_type

        data_module = EmbeddingDataModule(cfg)
        results = evaluate(cfg, data_module)

        # Save results to CSV using the utility function
        save_results_to_csv(results, cfg)

        print(f"Results for {task_name}:")
        print(
            f"Learning rate: {lr}, Batch size: {batch_size}, Max epochs: {max_epochs}"
        )
        print(results)
        print("--------------------")


if __name__ == "__main__":
    main()
