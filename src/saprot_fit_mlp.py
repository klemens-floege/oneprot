from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import ast

import tqdm
import h5py
import hydra
from omegaconf import DictConfig
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset



from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score

from scipy.stats import spearmanr


import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits
    


class MLPClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_labels, learning_rate=1e-3, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.model = MLP(input_size, hidden_size, num_labels, dropout_rate)
        self.learning_rate = learning_rate
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCEWithLogitsLoss()  # Change to BCEWithLogitsLoss
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)
        self.test_logits = []  # Add this line



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float()  # Ensure targets are of type Long
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float()  # Ensure targets are of type Long
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(logits, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float()  # Ensure targets are of type Long
        loss = self.loss_fn(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(logits, y))
        self.test_logits.append(logits)  # Collect logits during test
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_test_epoch_start(self):
        self.test_logits = []  # Reinitialize as an empty list at the start of the test epoch
    
    def on_test_epoch_end(self):
        self.test_logits = torch.cat(self.test_logits, dim=0)  # Concatenate logits at the end of testing

    
def get_dataloaders(train_data, val_data, test_data, batch_size=32):
    train_x, train_y = train_data
    val_x, val_y = val_data
    test_x, test_y = test_data

    train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_x, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def count_f1_max(pred, target): 
	"""
	    F1 score with the optimal threshold, Copied from TorchDrug.

	    This function first enumerates all possible thresholds for deciding positive and negative
	    samples, and then pick the threshold with the maximal F1 score.

	    Parameters:
	        pred (Tensor): predictions of shape :math:`(B, N)`
	        target (Tensor): binary targets of shape :math:`(B, N)` 
    """
    
	order = pred.argsort(descending=True, dim=1)
	target = target.gather(1, order)
	precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
	recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
	is_start = torch.zeros_like(target).bool()
	is_start[:, 0] = 1
	is_start = torch.scatter(is_start, 1, order, is_start)

	
	all_order = pred.flatten().argsort(descending=True)
	order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
	order = order.flatten()
	inv_order = torch.zeros_like(order)
	inv_order[order] = torch.arange(order.shape[0], device=order.device)
	is_start = is_start.flatten()[all_order]
	all_order = inv_order[all_order]
	precision = precision.flatten()
	recall = recall.flatten()
	all_precision = precision[all_order] - \
	                torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
	all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
	all_recall = recall[all_order] - \
	             torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
	all_recall = all_recall.cumsum(0) / pred.shape[0]
	all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
	return all_f1.max()

def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    print(cfg)
    output_dir = Path(cfg.output_dir) / cfg.task_name
    all_inputs = dict()

    for partition in cfg.evaluate_on:
        partition_dir = output_dir / partition
        seq_embs_list = []
        mod_embs_list = []
        targets_list = []
        
        # Iterate through subfolders in the partition directory
        for k, subfolder in tqdm.tqdm(enumerate(partition_dir.iterdir())):
            if subfolder.is_dir():
                file_range = subfolder.glob("*.npy")
                file_range = [int(f.stem.split("_")[0]) for f in file_range]
                file_range = sorted(list(set(file_range)))
                for idx in file_range:
                    seq_embs_list.append(np.load(subfolder / f"{idx}_seq.npy"))
                    if cfg.fit_classifier_on in ["struct", "both"]:
                        mod_embs_list.append(np.load(subfolder / f"{idx}_mod.npy"))
                    targets_list.append(np.load(subfolder / f"{idx}_target.npy"))

        # Concatenate embeddings if there are multiple files
        seq_embs = np.concatenate(seq_embs_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        if cfg.fit_classifier_on in ["struct", "both"]:
            mod_embs = np.concatenate(mod_embs_list, axis=0)
            print(partition, seq_embs.shape, mod_embs.shape, targets.shape)
        else:
            print(partition, seq_embs.shape, targets.shape)


        if cfg.fit_classifier_on == "seq":
            embeddings = seq_embs
        elif cfg.fit_classifier_on == "struct":
            embeddings = mod_embs
        elif cfg.fit_classifier_on == "both":
            embeddings = np.concatenate([seq_embs, mod_embs], axis=1)
        else:
            raise NotImplementedError

        # Convert targets to numeric values if they are in string format
        if isinstance(targets[0], str):
            targets = np.array([ast.literal_eval(t) for t in targets], dtype=np.float64)
        else:
            targets = np.array(targets, dtype=np.float64)

        # Ensure targets are one-dimensional or retain their original shape if they are one-hot encoded
        if targets.ndim == 1:
            targets = targets.squeeze()
        else:
            targets = targets
        
        
        # Convert targets to integer labels
        targets = targets.astype(int)

        print(f"Shape of embeddings: {embeddings.shape}")
        print(f"Shape of targets: {targets.shape}")

        # Create a mask to filter out rows with NaN values in embeddings and targets
        nan_mask_embeddings = ~np.isnan(embeddings).any(axis=1)
        nan_mask_targets = ~np.isnan(targets)

        #nan_mask = nan_mask_embeddings & nan_mask_targets

        # Apply the mask to filter out rows with NaN values
        #embeddings = embeddings[nan_mask]
        #targets = targets[nan_mask]

        

        all_inputs[f"{partition}_emb"] = embeddings
        all_inputs[f"{partition}_target"] = targets

    task_type = None

    train_data = (all_inputs["train_emb"], all_inputs["train_target"])
    val_data = (all_inputs["val_emb"], all_inputs["val_target"])
    test_data = (all_inputs["test_emb"], all_inputs["test_target"])

    train_loader, val_loader, test_loader = get_dataloaders(train_data, val_data, test_data, batch_size=cfg.classifier.batch_size)


    if cfg.task_name in ["MetalIonBinding", "GO_MF", "GO_CC", "GO_BP", "EC", "DeepLoc_cls2", "DeepLoc_cls10", "HumanPPI"]:
        task_type = 'classification'

        if cfg.task_name in ["GO_BP", "GO_MF", "GO_CC", "EC"]:

            # Instantiate model
            input_size = train_data[0].shape[1]
            hidden_size = cfg.classifier.hidden_size
            label2num = {"EC": 585, "GO_BP": 1943, "GO_MF": 489, "GO_CC": 320}
            num_labels = label2num[cfg.task_name]
            learning_rate =cfg.classifier.learning_rate

            classifier = MLPClassifier(input_size, hidden_size, num_labels, learning_rate=learning_rate)
            
            # Train the model
            
            trainer = pl.Trainer(
                max_epochs=cfg.classifier.max_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1 if torch.cuda.is_available() else None
            )
            trainer.fit(classifier, train_loader, val_loader)
    

        else: 
            classifier = hydra.utils.instantiate(cfg.classifier)
            classifier.fit(all_inputs["train_emb"], all_inputs["train_target"])

    elif cfg.task_name in ["Thermostability"]:
        task_type = 'regression'
        regressor = hydra.utils.instantiate(cfg.regressor)
        regressor.fit(all_inputs["train_emb"], all_inputs["train_target"])
    else:
        print("Task name not configured")
        raise Exception

    
    # train the classifier
    if task_type == 'classification':

        results = {}

        for partition in ["val", "test"]:
            print(f"Inference on {partition}...")
            

            if cfg.task_name in ["GO_BP", "GO_MF", "GO_CC", "EC"]:
                
                trainer.test(classifier, dataloaders=test_loader if partition == 'test' else val_loader)


                y_pred_logits = classifier.test_logits  # Extract logits
                #y_pred_proba = torch.softmax(torch.tensor(y_pred_logits), dim=1) # Convert logits to probabilities
                y_pred_proba = torch.sigmoid(torch.tensor(y_pred_logits))  # Correct for multi-label

                _pred_proba = torch.softmax(y_pred_logits, dim=1).cpu().numpy()  # Move logits to CPU and convert to probabilities
                

                label2num = {"EC": 585, "GO_BP": 1943, "GO_MF": 489, "GO_CC": 320}
                num_classes = label2num[cfg.task_name]
  

                y_pred_tensor = torch.tensor(y_pred_proba)
                print("y_pred_tensor shape ", y_pred_tensor.shape)
                
                target_tensor = torch.tensor(all_inputs[f"{partition}_target"])  # Add batch dimension
                # Find unique elements and their count
                unique_elements = torch.unique(target_tensor)
                num_unique_elements = unique_elements.size(0)
                max_element = torch.max(target_tensor)
                print(f"Number of unique elements: {num_unique_elements}")
                print(f"Maximum element: {max_element.item()}")

                print("y_pred_tensor ", y_pred_tensor[:5])
                print("target_tensor ", target_tensor[:5])

                #target_tensor_one_hot = F.one_hot(target_tensor, num_classes=num_classes).float()

                # Ensure correct reshaping for count_f1_max
                y_pred_reshaped = y_pred_tensor.view(-1, num_classes)
                target_reshaped = target_tensor.view(-1, num_classes)

                 # Move tensors to the same device (CPU) before calling count_f1_max
                y_pred_reshaped = y_pred_reshaped.cpu()
                target_reshaped = target_reshaped.cpu()


                print("y_pred_tensor shape ", y_pred_reshaped.shape)
                print("target_tensor shape ", target_reshaped.shape)
                print("y_pred_tensor ", y_pred_reshaped[:5])
                print("target_tensor ", target_reshaped[:5])
                #score = count_f1_max(torch.tensor(y_pred).view(1,-1), torch.tensor(all_inputs[f"{partition}_target"]).view(1,-1))
                score = count_f1_max(y_pred_reshaped, target_reshaped)
                print(f"{partition} Fmax Score: {score.item():.4f}")
                results[f"{partition}_fmax"] = score.item()

            else:
                y_pred = classifier.predict(all_inputs[f"{partition}_emb"])
                # Evaluate classifier
                accuracy = accuracy_score(all_inputs[f"{partition}_target"], y_pred)
                f1_micro = f1_score(all_inputs[f"{partition}_target"], y_pred, average='micro')
                f1_macro = f1_score(all_inputs[f"{partition}_target"], y_pred, average='macro')
                conf_matrix = confusion_matrix(all_inputs[f"{partition}_target"], y_pred)

                print(f"{partition} Accuracy: {accuracy:.4f}")
                print(f"{partition} F1 Score (Micro): {f1_micro:.4f}")
                print(f"{partition} F1 Score (Macro): {f1_macro:.4f}")
                print(f"{partition} Confusion Matrix:\n{conf_matrix}")


                #misclassified_idx = [i for i, (true, pred) in enumerate(zip(all_inputs[f"{partition}_target"], y_pred)) if true != pred]
                #print(f"{partition} Misclassified indices: {misclassified_idx}")

                results[f"{partition}_accuracy"] = accuracy
                results[f"{partition}_f1_micro"] = f1_micro
                results[f"{partition}_f1_macro"] = f1_macro
                results[f"{partition}_conf_matrix"] = conf_matrix

   

    else: 
        for partition in ["val", "test"]:
            print(f"Inference on {partition}...")
            y_pred = regressor.predict(all_inputs[f"{partition}_emb"])

            mse = mean_squared_error(all_inputs[f"{partition}_target"], y_pred)
            r2 = r2_score(all_inputs[f"{partition}_target"], y_pred)
            spearman_rho, _ = spearmanr(all_inputs[f"{partition}_target"], y_pred)

            print(f"{partition} Mean Squared Error: {mse:.4f}")
            print(f"{partition} R2 Score: {r2:.4f}")
            print(f"{partition} Spearman's Rank Correlation Coefficient: {spearman_rho:.4f}")



@hydra.main(version_base="1.3", config_path="../configs", config_name="saprot.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    print('taskname: ', cfg.task_name)

    # Access the classifier configuration
    classifier_cfg = cfg.classifier

    # Example: Print classifier config to ensure it's loaded
    print("Classifier config:", classifier_cfg)


    evaluate(cfg)


if __name__ == "__main__":
    main()
