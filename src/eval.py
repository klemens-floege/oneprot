import os
from typing import Any, List, Tuple

import tqdm
import faiss
import hydra
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchmetrics.metric import Metric 
from torchmetrics.utilities.data import dim_zero_cat
from torch_geometric.data import Batch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.datasets import MSADataset, StructDataset, PocketDataset, TextDataset

def identity_map(embedding, cfg):
    return embedding

def threshold_map(embedding, cfg):
    dev = embedding.device
    return torch.where(embedding > cfg.bit_threshold, torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev))

function_map = {
    "identity": identity_map,
    "threshold": threshold_map,
}


class FAISSMetric(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(self,  **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds.detach())
        self.target.append(target.detach())

    def compute(self) -> Tensor:
        pass

    def compute_no_cache(self, k=3) -> Tensor:
        seq_out = dim_zero_cat(self.preds).detach().cpu().numpy()
        mod_out = dim_zero_cat(self.target).detach().cpu().numpy()
        evaluation_targets = [("pred_to_target", seq_out, mod_out), ("target_to_pred", mod_out, seq_out)]
        metrics = dict()

        for name, input, target in evaluation_targets:
            index = faiss.IndexFlatL2(input.shape[1])
            index.add(input)
            _, I = index.search(target, k)
            correct = np.array([int(j in i) for j, i in enumerate(I)])
            metrics[name] =  correct.mean()
        return metrics


class JoinedDataset(Dataset):
    def __init__(self, csv_file: str, data_config: DictConfig):
        with open(csv_file, "r") as f:
            self.data = f.readlines()

        self.data = self.data[1:]
        self.data = [line.strip().split(",") for line in self.data]
        # workaround for inconsistent data format
        if str(csv_file.stem).lower()[:6] == "struct":
            self.data = [(line[0], line[0], line[1]) for line in self.data]
        modalities = str(csv_file.stem).lower().split("_")
        
        self.datasets = dict()
        self.modalities = modalities
        for i, modality in enumerate(modalities):
            if modality == "struct":
                self.datasets[modality] = StructDataset(data_dir=data_config.data_dir, split='test', seq_tokenizer=data_config.seq_tokenizer)
            elif modality == "msa":
                self.datasets[modality] = MSADataset(data_dir=data_config.data_dir, split='test', seq_tokenizer=data_config.seq_tokenizer)
            elif modality == "pocket":
                # self.datasets[modality] = PocketDataset(split='test', seq_tokenizer=data_config.seq_tokenizer, data_type=data_config.pocket_data_type)
                self.datasets[modality] =  StructDataset(data_dir=data_config.data_dir, split='test', seq_tokenizer=data_config.seq_tokenizer, pockets=True, seqsim='30ss')
            elif modality == "text":
                self.datasets[modality] = TextDataset(data_dir=data_config.data_dir, split='test', seq_tokenizer=data_config.seq_tokenizer, text_tokenizer=data_config.text_tokenizer)
                self.dummy_struct_ds = StructDataset(data_dir=data_config.data_dir, split='test', seq_tokenizer=data_config.seq_tokenizer)
            else:
                raise ValueError(f"Unknown modality {modality}")
            
 
    def __len__(self):
        return len(self.data)
       
    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        uniprot_ids = [d[0] for d in data]
        modality_data = [d[1:] for d in data]

        output_batch = dict(uniprot_ids=uniprot_ids)
        for k, mod in enumerate(self.modalities):
            current_modality_data = [d[k] for d in modality_data]
            if mod == "text":
                # we cheat a bit for the sequences that belong to text, we just get them from the structure module :) 
                # (text doesnt allow access out of the box)
                sequences = self.dummy_struct_ds.collate_fn(uniprot_ids, return_raw_sequences=True)
                # we can probably solve this nicer...
                current_modality_data = [text.strip("\"").strip("[").strip("]").strip("\\").strip("\'") for text in current_modality_data]
                current_modality_data = list(zip(current_modality_data, sequences))

            modality_batch = self.datasets[mod].collate_fn(current_modality_data)
            modality_batch = move_batch_to_device(modality_batch)
            output_batch[mod] = modality_batch
            
        return output_batch
    

def build_dataloader(cfg, dataset):
    # workaround for Pocket Dataset pickling error
    num_workers = 0 if "pocket" in dataset.modalities else int(os.getenv('SLURM_CPUS_PER_TASK'))
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        drop_last=False,
    )

def move_batch_to_device(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(batch, tuple):
        batch = list(batch)
        if isinstance(batch[1], dict):
            batch[1] = {key: value.to(device) if (torch.is_tensor(value) or isinstance(value, Batch)) else value for key, value in batch[1].items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, list):
        return [item.to(device) if (torch.is_tensor(item) or isinstance(item, Batch)) else item for item in batch]
    else:
        raise TypeError

def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    model = hydra.utils.instantiate(cfg.model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])
        model.cuda()
    else:
        model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    

    modalities = model.data_modalities
    modality_files = [file for file in os.listdir(cfg.retrieval_dataset) if file.endswith(".csv")]
    data_file_path = Path(cfg.retrieval_dataset)
    
    with torch.no_grad():
        all_retrieval_results = dict()
        for modality_file in modality_files:
            modality_tuple = modality_file.lower().split(".")[0].split("_")
    
            if len(set(modality_tuple) & set(modalities)) != 2:
                continue
            
            dataset = JoinedDataset(data_file_path /  modality_file, cfg.data)
            dataloader = build_dataloader(cfg.data, dataset)
            
            metric_dict = {
                f"{modality_tuple[0]}_{modality_tuple[1]}": FAISSMetric(), 
                f"seq_{modality_tuple[0]}": FAISSMetric(), 
                f"seq_{modality_tuple[1]}": FAISSMetric()}
            
            for i, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Collecting {modality_tuple[0]}, {modality_tuple[1]} embeddings")):
                modality_embeddings = []
                for modality in modality_tuple:
                    seq, embs = model(*batch[modality], modality=modality)
                    seq = function_map[cfg.transform_func](seq, cfg)
                    embs = function_map[cfg.transform_func](embs, cfg)
                    modality_embeddings.append(embs)
                    metric_dict[f"seq_{modality}"].update(seq, embs)
                metric_dict[f"{modality_tuple[0]}_{modality_tuple[1]}"].update(*modality_embeddings)
    
            retrieval_results = dict()
            for key in metric_dict:
                for k in cfg.eval_ks:
                    results = metric_dict[key].compute_no_cache(k=k)
                    key_parts = key.split("_")
                    retrieval_results[f"{key_parts[0]}_to_{key_parts[1]}@{k}"] = results["pred_to_target"]
                    retrieval_results[f"{key_parts[1]}_to_{key_parts[0]}@{k}"] = results["target_to_pred"]
            print(retrieval_results)
            all_retrieval_results.update(retrieval_results)
    return all_retrieval_results



@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    evaluate(cfg)


if __name__ == "__main__":
    main()
