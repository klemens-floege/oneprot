from typing import Dict
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

import sys
import os


# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory and its parent to the Python path
sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from src.utils.downstream_utils import save_results_to_csv, load_data


def evaluate(cfg: DictConfig, all_inputs: Dict[str, np.ndarray]) -> Dict:
    model = hydra.utils.instantiate(cfg.downstream_model)

    model.fit(all_inputs["train_emb"], all_inputs["train_target"])

    results = {}
    for partition in ["valid", "test"]:
        y_pred = model.predict(all_inputs[f"{partition}_emb"])
        mse = mean_squared_error(all_inputs[f"{partition}_target"], y_pred)
        r2 = r2_score(all_inputs[f"{partition}_target"], y_pred)
        spearman_rho, _ = spearmanr(all_inputs[f"{partition}_target"], y_pred)

        results[f"{partition}_mse"] = mse
        results[f"{partition}_r2"] = r2
        results[f"{partition}_spearman_rho"] = spearman_rho

    return results


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="saprot_sweep_xgboost_reg.yaml",
)
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    all_inputs = load_data(cfg)
    results = evaluate(cfg, all_inputs)
    save_results_to_csv(results, cfg)


if __name__ == "__main__":
    main()
