from typing import Dict
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import sys
import os

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory and its parent to the Python path
sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.utils.downstream import save_results_to_csv, load_data, count_f1_max
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

def evaluate(cfg: DictConfig, all_inputs: Dict[str, np.ndarray]) -> Dict:
    if cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI"]:
        cfg.downstream_model.objective = "binary:logistic"
    elif cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
        cfg.downstream_model.objective = "binary:logistic"
    elif cfg.task_name in ["ThermoStability"]:
        cfg.downstream_model.objective = "reg:squarederror"
    else:
        cfg.downstream_model.objective = "multi:softmax"
    
    model = hydra.utils.instantiate(cfg.downstream_model)

    model.fit(all_inputs["train_emb"], all_inputs["train_target"])

    results = {}
    for partition in ["valid", "test"]:
        y_pred = model.predict(all_inputs[f"{partition}_emb"])
        y_true = all_inputs[f"{partition}_target"]

        if cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI"]:
            accuracy = accuracy_score(y_true, y_pred > 0.5)
            f1_micro = f1_score(y_true, y_pred > 0.5, average="micro")
            auc = roc_auc_score(y_true, y_pred)
            results[f"{partition}_accuracy"] = accuracy
            results[f"{partition}_f1_micro"] = f1_micro
            results[f"{partition}_auc"] = auc

        elif cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC"]:
            f1_max = count_f1_max(y_pred, y_true)
            results[f"{partition}_f1_max"] = f1_max

        elif cfg.task_name in ["ThermoStability"]:
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            spearman_rho, _ = spearmanr(y_true, y_pred)
            results[f"{partition}_mse"] = mse
            results[f"{partition}_r2"] = r2
            results[f"{partition}_spearman_rho"] = spearman_rho

        else:  # multi_class
            accuracy = accuracy_score(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average="micro")
            results[f"{partition}_accuracy"] = accuracy
            results[f"{partition}_f1_micro"] = f1_micro

    return results

@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="saprot_sweep_xgboost_cls.yaml",
)
def main(cfg: DictConfig) -> None:
    all_inputs = load_data(cfg)
    results = evaluate(cfg, all_inputs)
    save_results_to_csv(results, cfg)

if __name__ == "__main__":
    main()