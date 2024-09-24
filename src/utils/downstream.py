import torch
import pandas as pd
import numpy as np
import csv
from typing import Dict
from pathlib import Path
from omegaconf import DictConfig
import os
import math


def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold, Copied from TorchDrug.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    print(f"Pred type: {type(pred)}, Target type: {type(target)}")
    print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max().item()


def save_results_to_csv(results: Dict, cfg: DictConfig) -> None:
    output_dir = Path(cfg.results_dir) / "downstream_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{cfg.task_name}_{cfg.downstream_model.name}_results.csv"

    results_df = pd.DataFrame([results])
    results_df["model_type"] = cfg.model_type
    results_df["downstream_model"] = cfg.downstream_model.name
    results_df["task_name"] = cfg.task_name
    results_df["threshold"] = cfg.threshold

    # Add hyperparameters as separate columns
    for param, value in cfg.downstream_model.items():
        if param != "_target_":
            results_df[f"param_{param}"] = value

    first_columns = ["model_type"] + [
        col for col in results_df.columns if col.startswith("test_")
    ]

    # Get the remaining columns
    remaining_columns = [col for col in results_df.columns if col not in first_columns]

    # Reorder the DataFrame columns
    results_df = results_df[first_columns + remaining_columns]
    # Define a fixed width for each column
    column_width = 20

    # Function to format values (rounding numerics, handling strings)
    def format_value(value):
        if isinstance(value, (int, float, np.number)):
            return f"{value:.3f}".ljust(column_width)
        else:
            return str(value).ljust(column_width)

    # Write to CSV with fixed column widths
    mode = "a" if csv_path.exists() else "w"

    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if mode == "w":
            header = results_df.columns.tolist()
            writer.writerow([col.ljust(column_width) for col in header])

        # Write column names and data
        for _, row in results_df.iterrows():
            formatted_row = [format_value(value) for value in row]
            writer.writerow(formatted_row)

    print(f"Results appended to {csv_path}")

    # Display the results in a formatted table
    print("\nResults:")
    for col, value in zip(results_df.columns, results_df.iloc[0]):
        print(f"{col.ljust(30)}: {format_value(value)}")


def load_data(cfg: DictConfig) -> Dict[str, np.ndarray]:
    all_inputs = {}
    for partition in cfg.evaluate_on:
        embedding_path = os.path.join(
            cfg.emb_dir,
            f"{cfg.model_type}/{cfg.task_name}/{partition}/{cfg.task_name}_{partition}_embeddings_labels.pt",
        )
        if os.path.exists(embedding_path) and os.path.exists(embedding_path):
            print(f"Loading Embeddings From {embedding_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            embeddings_w_target = torch.load(embedding_path, map_location=device)
            embeddings = embeddings_w_target["embeddings"]
            targets = embeddings_w_target["labels_fitness"]
            print(
                f"Embeddins Size {embeddings.size()[0], embeddings.size()[1]}, Targets Size {targets.size()[0]}"
            )

        if not math.isinf(cfg.threshold):
            embeddings = torch.where(
                embeddings > cfg.threshold, torch.tensor(1.0), torch.tensor(0.0)
            )

        all_inputs[f"{partition}_emb"] = embeddings.cpu().numpy()
        all_inputs[f"{partition}_target"] = targets.cpu().numpy()

    return all_inputs
