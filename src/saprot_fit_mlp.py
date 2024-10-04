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

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256], dropout_rate=0.0, 
                 use_batch_norm=False, use_layer_norm=False, activation='relu', 
                 use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.input_dim = input_dim

        layers = []
        in_features = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_features, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_features, output_dim)

        # Add a projection layer for residual connections if needed
        if self.use_residual and input_dim != hidden_dims[-1]:
            self.residual_projection = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual_projection = None

    def forward(self, x):
        if self.use_residual:
            residual = x

        for layer in self.hidden_layers:
            x = layer(x)

        if self.use_residual:
            if self.residual_projection:
                residual = self.residual_projection(residual)
            x = x + residual

        return self.output_layer(x)

class EvaluationModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        if cfg.model_type == "esm2":
            input_dim = 1280
        elif cfg.model_type in ["oneprot_9"]:
            input_dim = 256
        elif cfg.model_type in ["oneprot_15", "oneprot_16"]:
            input_dim = 1280
        else:
            input_dim = 512
        if self.cfg.task_name == "HumanPPI":
            input_dim = input_dim * 2

        # Determine output_dim based on task_name
        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI", "ThermoStability"]:
            output_dim = 1
        elif self.cfg.task_name == "EC":
            output_dim = 585
        elif self.cfg.task_name == "GO-BP":
            output_dim = 1943
        elif self.cfg.task_name == "GO-MF":
            output_dim = 489
        elif self.cfg.task_name == "GO-CC":
            output_dim = 320
        elif self.cfg.task_name == "DeepLoc10":
            output_dim = 10
        else:
            raise ValueError(f"Unknown task_name: {self.cfg.task_name}")

        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.model.hidden_dims,
            dropout_rate=cfg.model.dropout_rate,
            use_batch_norm=cfg.model.use_batch_norm,
            use_layer_norm=cfg.model.use_layer_norm,
            activation=cfg.model.activation,
            use_residual=cfg.model.use_residual
        )

        # Set loss function based on task_name
        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI", "EC", "GO-BP", "GO-MF", "GO-CC"]:
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif self.cfg.task_name in ["ThermoStability"]:
            self.loss_fn = F.mse_loss
        else:  # multi_class
            self.loss_fn = F.cross_entropy

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

        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI"]:
            y_hat = y_hat.squeeze(1)
            preds = torch.sigmoid(y_hat)
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
            preds = torch.sigmoid(y_hat)
        elif self.cfg.task_name in ["ThermoStability"]:
            preds = y_hat.squeeze(1)
        else:  # multi_class
            preds = torch.softmax(y_hat, dim=1)
            preds = torch.argmax(preds, dim=1)

        return preds, y

def evaluate(cfg: DictConfig, data_module: EmbeddingDataModule) -> Dict:
    model = EvaluationModule(cfg)

    # Determine the accelerator and devices based on GPU availability
    if torch.cuda.is_available():
        accelerator = "gpu"
        fit_devices = 4
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
            accuracy = accuracy_score(y_true, y_pred > 0.5)
            f1_micro = f1_score(y_true, y_pred > 0.5, average="micro")
            auc = roc_auc_score(y_true, y_pred)
            results[f"{partition}_accuracy"] = accuracy
            results[f"{partition}_f1_micro"] = f1_micro
            results[f"{partition}_auc"] = auc

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
            results[f"{partition}_mse"] = mse
            results[f"{partition}_r2"] = r2
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
        cfg.sweep.hidden_dims,
        cfg.sweep.dropout_rate,
        cfg.sweep.use_batch_norm,
        cfg.sweep.use_layer_norm,
        cfg.sweep.activation,
        cfg.sweep.use_residual,
        cfg.sweep.task_name,
        cfg.sweep.model_type,
    )

    for (
        lr,
        batch_size,
        max_epochs,
        hidden_dims,
        dropout_rate,
        use_batch_norm,
        use_layer_norm,
        activation,
        use_residual,
        task_name,
        model_type,
    ) in param_combinations:
        # Update the configuration with the current hyperparameters
        cfg.model.learning_rate = lr
        cfg.model.batch_size = batch_size
        cfg.model.max_epochs = max_epochs
        cfg.model.hidden_dims = hidden_dims
        cfg.model.dropout_rate = dropout_rate
        cfg.model.use_batch_norm = use_batch_norm
        cfg.model.use_layer_norm = use_layer_norm
        cfg.model.activation = activation
        cfg.model.use_residual = use_residual
        cfg.task_name = task_name
        cfg.model_type = model_type

        data_module = EmbeddingDataModule(cfg)
        results = evaluate(cfg, data_module)

        # Save results to CSV using the utility function
        save_results_to_csv(results, cfg)

        print(f"Results for {task_name}:")
        print(f"Learning rate: {lr}, Batch size: {batch_size}, Max epochs: {max_epochs}")
        print(f"Hidden dims: {hidden_dims}, Dropout rate: {dropout_rate}")
        print(f"Use batch norm: {use_batch_norm}, Use layer norm: {use_layer_norm}")
        print(f"Activation: {activation}, Use residual: {use_residual}")
        print(f"Model type: {model_type}")
        print(results)
        print("--------------------")


if __name__ == "__main__":
    main()