import os
from typing import Any, List, Tuple, Dict

import tqdm
import hydra
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import dataset classes
from src.data.datasets import MSADataset, StructDataset, PocketDataset, TextDataset

def identity_map(embedding: Tensor, cfg: DictConfig) -> Tensor:
    """Identity function for embeddings."""
    return embedding

def threshold_map(embedding: Tensor, cfg: DictConfig) -> Tensor:
    """Apply threshold to embeddings, converting them to binary."""
    return torch.where(embedding > cfg.bit_threshold, 
                       torch.tensor(1.0, device=embedding.device), 
                       torch.tensor(0.0, device=embedding.device))

# Mapping of transformation function names to their implementations
function_map = {
    "identity": identity_map,
    "threshold": threshold_map,
}

def build_dataloader(cfg: DictConfig, dataset: Dataset) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        cfg (DictConfig): Configuration object containing dataloader parameters.
        dataset (Dataset): The dataset to create a DataLoader for.

    Returns:
        DataLoader: The created DataLoader object.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        drop_last=False,
    )

def evaluate(cfg: DictConfig) -> Dict[str, float]:
    """
    Main evaluation function for cross-modality retrieval.

    Args:
        cfg (DictConfig): Configuration object containing evaluation parameters.

    Returns:
        Dict[str, float]: Dictionary containing retrieval results.
    """
    # Load the model from checkpoint
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(cfg.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()

    modalities = model.modalities
    data_file_path = Path(cfg.retrieval_dataset)
    
    all_embeddings = {}
    all_sequences = {}

    # Compute embeddings for each modality
    with torch.no_grad():
        for modality in modalities:
            dataset = create_dataset(modality, data_file_path, cfg.data)
            dataloader = build_dataloader(cfg, dataset)
            
            embeddings = []
            sequences = []
            
            for batch in tqdm.tqdm(dataloader, desc=f"Embedding {modality}"):
                seq_inputs, mod_inputs, _, batch_sequences = batch
                seq_features, mod_features = model(seq_inputs, mod_inputs, modality=modality)
                
                # Apply transformation function (identity or threshold)
                seq_features = function_map[cfg.transform_func](seq_features, cfg)
                mod_features = function_map[cfg.transform_func](mod_features, cfg)
                
                embeddings.append(mod_features)
                sequences.extend(batch_sequences)
            
            # Store embeddings and sequences for each modality
            all_embeddings[f"sequence_{modality}"] = torch.cat(embeddings)
            all_embeddings[modality] = torch.cat(embeddings)
            all_sequences[modality] = sequences

    # Compute cross-modality retrieval metrics
    retrieval_results = compute_cross_modality_metrics(all_embeddings, all_sequences, cfg.eval_ks)
    
    print(retrieval_results)
    return retrieval_results

def create_dataset(modality: str, data_path: Path, data_config: DictConfig) -> Dataset:
    """
    Create a dataset object for the given modality.

    Args:
        modality (str): The modality to create a dataset for.
        data_path (Path): Path to the data directory.
        data_config (DictConfig): Configuration object for dataset creation.

    Returns:
        Dataset: The created dataset object.

    Raises:
        ValueError: If an unknown modality is provided.
    """
    if modality == "struct_graph":
        return StructDataset(data_dir=data_path, split='test', seq_tokenizer=data_config.seq_tokenizer)
    if modality == "struct_token":
        return StructDataset(data_dir=data_path, split='test', seq_tokenizer=data_config.seq_tokenizer)
    elif modality == "msa":
        return MSADataset(data_dir=data_path, split='test', seq_tokenizer=data_config.seq_tokenizer)
    elif modality == "pocket":
        return StructDataset(data_dir=data_path, split='test', seq_tokenizer=data_config.seq_tokenizer, pockets=True, seqsim='30ss')
    elif modality == "text":
        return TextDataset(data_dir=data_path, split='test', seq_tokenizer=data_config.seq_tokenizer, text_tokenizer=data_config.text_tokenizer)
    else:
        raise ValueError(f"Unknown modality {modality}")

def compute_cross_modality_metrics(embeddings: Dict[str, Tensor], sequences: Dict[str, List[str]], k_values: List[int]) -> Dict[str, float]:
    """
    Compute cross-modality retrieval metrics for all pairs of modalities.

    Args:
        embeddings (Dict[str, Tensor]): Dictionary of embeddings for each modality.
        sequences (Dict[str, List[str]]): Dictionary of sequences for each modality.
        k_values (List[int]): List of k values for top-k retrieval evaluation.

    Returns:
        Dict[str, float]: Dictionary of retrieval results for all modality pairs and k values.
    """
    results = {}
    modalities = [mod for mod in embeddings.keys() if not mod.startswith("sequence_")]
    
    # Compute metrics for sequence-to-modality and modality-to-sequence
    for mod in modalities:
        seq_mod = f"sequence_{mod}"
        results.update(compute_retrieval_metrics(f"sequence_to_{mod}", embeddings[seq_mod], embeddings[mod], k_values))
        results.update(compute_retrieval_metrics(f"{mod}_to_sequence", embeddings[mod], embeddings[seq_mod], k_values))

    # Compute metrics for modality-to-modality
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i+1:]:
            results.update(compute_retrieval_metrics(f"{mod1}_to_{mod2}", embeddings[mod1], embeddings[mod2], k_values))
            results.update(compute_retrieval_metrics(f"{mod2}_to_{mod1}", embeddings[mod2], embeddings[mod1], k_values))
    
    return results

def compute_retrieval_metrics(name: str, query_embeddings: Tensor, gallery_embeddings: Tensor, k_values: List[int]) -> Dict[str, float]:
    """
    Compute retrieval metrics for a pair of query and gallery embeddings.

    Args:
        name (str): Name of the retrieval task (e.g., "sequence_to_structure").
        query_embeddings (Tensor): Embeddings of the query items.
        gallery_embeddings (Tensor): Embeddings of the gallery items.
        k_values (List[int]): List of k values for top-k retrieval evaluation.

    Returns:
        Dict[str, float]: Dictionary of retrieval results for different k values.
    """
    results = {}
    # Compute similarity matrix
    sim_matrix = torch.matmul(query_embeddings, gallery_embeddings.t())
    
    for k in k_values:
        # Get top-k indices
        top_k = torch.topk(sim_matrix, min(k, sim_matrix.shape[1]), dim=1)[1]
        # Check if the correct match is in the top-k
        correct = (top_k == torch.arange(sim_matrix.shape[0]).unsqueeze(1).to(top_k.device)).any(dim=1)
        # Compute accuracy (Recall@k)
        accuracy = correct.float().mean().item()
        results[f"{name}@{k}"] = accuracy
    return results

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the evaluation script.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    evaluate(cfg)

if __name__ == "__main__":
    main()