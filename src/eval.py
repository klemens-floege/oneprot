import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import csv
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import pandas as pd
import esm
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer
import pyrootutils
import h5py
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.utils import task_wrapper, instantiate_callbacks, instantiate_loggers, log_hyperparameters, extras, get_pylogger
from src.data.utils.msa_utils import read_msa, filter_and_create_msa_file_list, greedy_select
from src.data.utils.struct_graph_utils import protein_to_graph

logger = logging.getLogger(__name__)

class CombinedDataset(Dataset):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.dataset
        
        self.column_names = [
            'ids', 'msa_files', 'text', 'struct_token', 
            'struct_graph', 'sequence', 'pocket'
        ]
        
        self.data = pd.read_csv(self.cfg.csv_file_path, header=None, names=self.column_names)
        self.data.drop(self.data.index[0], inplace=True)
        print(self.data.head())
      
        self.pocket_h5_file = f'{self.cfg.data_dir}/pockets_100_residues.h5'
        self.struct_h5_file = f'{self.cfg.data_dir}/seqstruc.h5'

        self.msa_files = filter_and_create_msa_file_list(self.cfg.csv_file_path)
        _, msa_transformer_alphabet = esm.pretrained.load_model_and_alphabet_local(self.cfg.msa_model_name_or_path)
        self.msa_padding_idx = 1
        self.msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=self.cfg.max_length)

        self.sequence_tokenizer = self._initialize_tokenizer(self.cfg.seq_tokenizer)
        self.structure_tokenizer = self._initialize_tokenizer(self.cfg.seq_tokenizer, add_tokens=True)
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_tokenizer)

    def _initialize_tokenizer(self, tokenizer_name: str, add_tokens: bool = False) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if add_tokens:
            new_tokens = ['p', 'y', 'n', 'w', 'r', 'q', 'h', 'g', 'd', 'l', 'v', 't', 'm', 'f', 's', 'a', 'e', 'i', 'k', 'c', '#']
            tokenizer.add_tokens(new_tokens)
        return tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, str]:
        item = self.data.iloc[index].to_dict()
        return item

    def collate_fn(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        ids = [item['ids'] for item in batch]
        func_texts = [item['text'] for item in batch]
        sequences_ids = [item['sequence'] for item in batch]
        structures = [item['struct_token'] for item in batch]
        struct_ids = [item['struct_graph'] for item in batch]
        pocket_ids = [item['pocket'] for item in batch]
        
        seq_inputs = []
        struct_inputs = []
        pocket_inputs = []
        for struct_id, pocket_id, seq_id in zip(struct_ids, pocket_ids, sequences_ids):
            try:
                with h5py.File(self.struct_h5_file, 'r') as file:
                    seq_inputs.append(file[seq_id]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8'))

                struct_inputs.append(protein_to_graph(struct_id, self.struct_h5_file, 'non_pdb', 'A', pockets=False))
                pocket_inputs.append(protein_to_graph(pocket_id, self.pocket_h5_file, 'non_pdb', 'A', pockets=True))
            except KeyError:
                logger.warning(f"KeyError: --{struct_id}-- or one of its pockets not found in h5 file")
        

        
        if self.cfg.remove_hash:
            structures = [s.replace("#", "") for s in structures]
        
        sequence_input = self._tokenize(self.sequence_tokenizer, seq_inputs)
        structure_input = self._tokenize(self.structure_tokenizer, structures)
        text_input = self.text_tokenizer(func_texts, max_length=self.cfg.text_max_length, padding=True, truncation=True, return_tensors="pt").input_ids   
        
        return {
            'ids': ids,
            'sequence': sequence_input,
            'struct_token': structure_input,
            'text': text_input,
            'struct_graph': Batch.from_data_list(struct_inputs),
            'pocket': Batch.from_data_list(pocket_inputs)
        }

    def _tokenize(self, tokenizer: AutoTokenizer, texts: List[str]) -> torch.Tensor:
        return tokenizer(
            texts, 
            max_length=self.cfg.max_sequence_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids

def load_custom_model(cfg: DictConfig) -> pl.LightningModule:
    logger.info("Loading custom model configuration")
    
    model_config_path = cfg.model.config_path
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")
    
    model_cfg = OmegaConf.load(model_config_path)
    
    logger.info("Instantiating custom model")
    model = hydra.utils.instantiate(model_cfg.model)

    if cfg.model.ckpt_path is not None:
        logger.info(f"Loading model checkpoint from: {cfg.model.ckpt_path}")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(cfg.model.ckpt_path)["state_dict"])
            model.cuda()
        else:
            model.load_state_dict(
                torch.load(cfg.model.ckpt_path, map_location="cpu")["state_dict"]
            )
    model.eval()
    logger.info("Custom model loaded successfully")
    return model

def create_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = CombinedDataset(cfg)
    return DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=cfg.dataloader.shuffle, 
                      num_workers=cfg.dataloader.num_workers, collate_fn=dataset.collate_fn)

def encode_inputs(model: pl.LightningModule, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model.eval()
    encoded_outputs = {}
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for modality, data in batch.items():
            if modality not in ['ids']:
                data = data.to(device)
                encoded_outputs[modality] = model(data, modality)
    
    return encoded_outputs

def calculate_retrieval_metrics(embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    modalities = list(embeddings.keys())
    results = {}
    
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i+1:]:
            emb1, emb2 = embeddings[mod1], embeddings[mod2]
            
            # Log dimensions of embedding matrices
            logger.info(f"Dimensions of {mod1} embedding: {emb1.shape}")
            logger.info(f"Dimensions of {mod2} embedding: {emb2.shape}")
            
            similarity_matrix = cosine_similarity(emb1, emb2)
            
            metrics = {}
            
            # Calculate metrics for both directions
            for name, logit in [("seq_to_mod", similarity_matrix), ("mod_to_seq", similarity_matrix.T)]:
                ranking = np.argsort(-logit, axis=1)
                preds = np.where(ranking == np.arange(len(logit))[:, None])[1]
                metrics[f"{name}_median_rank"] = int(np.floor(np.median(preds)) + 1)
                for k in [1, 10, 100, 500]:
                    metrics[f"{name}_R@{k}"] = np.mean(preds < k)
            
            results[f'{mod1}-{mod2}'] = metrics
    
    return results
def write_results_to_csv(results: Dict[str, Dict[str, float]], output_path: str):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        headers = ['Modality Pair           ', 'R@1        ', 'R@10       ', 'R@100      ', 'R@500      ', 'MR         ']
        writer.writerow(headers)
        
        for modality_pair, metrics in results.items():
            mod1, mod2 = modality_pair.split('-')
            for direction in ['seq_to_mod', 'mod_to_seq']:
                if direction == 'seq_to_mod':
                    pair = f"{mod1}-{mod2}"
                else:
                    pair = f"{mod2}-{mod1}"
                
                row = [
                    f"{pair:<25}",
                    f"{metrics[f'{direction}_R@1']:.3f}      ",
                    f"{metrics[f'{direction}_R@10']:.3f}      ",
                    f"{metrics[f'{direction}_R@100']:.3f}      ",
                    f"{metrics[f'{direction}_R@500']:.3f}      ",
                    f"{metrics[f'{direction}_median_rank']:<11}"
                ]
                writer.writerow(row)
@hydra.main(config_path="../configs", config_name="calculate_retrieval.yaml")
def main(cfg: DictConfig):
    # Load model using the provided function
    model = load_custom_model(cfg)
    
    # Create DataLoader
    dataloader = create_dataloader(cfg)
    
    # Check for available modalities
    available_modalities = list(next(iter(dataloader)).keys())
    logger.info(f"Available modalities: {available_modalities}")
    
    all_embeddings = {modality: [] for modality in available_modalities if modality != 'ids'}
    
    # Process dataset
    for batch in dataloader:
        embedded_tokens = encode_inputs(model, batch)
        for modality, embeddings in embedded_tokens.items():
            if modality != 'ids':
                all_embeddings[modality].append(embeddings.cpu().numpy())
    
    # Concatenate embeddings
    for modality in all_embeddings:
        all_embeddings[modality] = np.concatenate(all_embeddings[modality], axis=0)
        logger.info(f"Final dimensions of {modality} embeddings: {all_embeddings[modality].shape}")
    
    # Calculate retrieval metrics
    results = calculate_retrieval_metrics(all_embeddings)
    
    # Write results to CSV
    write_results_to_csv(results, cfg.output.csv_path)
    
    logger.info(f"Retrieval metrics calculation completed. Results written to {cfg.output.csv_path}")

if __name__ == "__main__":
    main()