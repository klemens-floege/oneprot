from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import tqdm
import h5py
import hydra
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)




def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    print(cfg)
    output_dir = Path(cfg.output_dir)
    all_inputs = dict()

    for partition in cfg.evaluate_on:
        partition_dir = output_dir / partition
        seq_embs_list = []
        mod_embs_list = []
        targets_list = []
        
        # load npy file
        path = f"/p/project/hai_oneprot/floege1/oneprot-2/{partition}_embeddings.npy"
        embeddings = np.load(path)

        print(partition, embeddings.shape)

        nan_mask = ~np.isnan(embeddings).any(axis=1) & ~np.isnan(targets)

        # Apply the mask to filter out rows with NaN values
        embeddings = embeddings[nan_mask]
        targets = targets[nan_mask]

        print(nan_mask.sum())

        all_inputs[f"{partition}_emb"] = embeddings
        all_inputs[f"{partition}_target"] = targets

    
    # train the classifier
    classifier = hydra.utils.instantiate(cfg.classifier)
    classifier.fit(all_inputs["train_emb"], all_inputs["train_target"])

    for partition in ["val", "test"]:
        print(f"Inference on {partition}...")
        y_pred = classifier.predict(all_inputs[f"{partition}_emb"])

        # evaluate classifier
        accuracy = accuracy_score(all_inputs[f"{partition}_target"], y_pred)
        f1_micro = f1_score(all_inputs[f"{partition}_target"], y_pred, average='micro')
        f1_macro = f1_score(all_inputs[f"{partition}_target"], y_pred, average='macro')
        conf_matrix = confusion_matrix(all_inputs[f"{partition}_target"], y_pred)

        print(f"{partition} Accuracy: {accuracy}") # accuracy should be the same as F1 Macro
        print(f"{partition} F1 Score (Micro): {f1_micro}")
        print(f"{partition} F1 Score (Macro): {f1_macro}")
        print(f"{partition} Confusion Matrix:\n{conf_matrix}")

        misclassified_idx = [i for i, (true, pred) in enumerate(zip(all_inputs[f"{partition}_target"], y_pred)) if true != pred]
        print(f"{partition} Misclassified indices: {misclassified_idx}")



@hydra.main(version_base="1.3", config_path="configs", config_name="downstream.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    evaluate(cfg)


if __name__ == "__main__":
    main()
