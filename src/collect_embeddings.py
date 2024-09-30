import os
import logging
from typing import List, Any
import shutil

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils
import ast

# Set up logging
if int(os.environ.get("SLURM_PROCID", 0)) == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
else:
    logging.basicConfig(level=logging.CRITICAL)

logger = logging.getLogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class SequenceDataset(Dataset):
    def __init__(self, csv_file: str, label_type: str = "classification"):
        logger.info(f"Initializing SequenceDataset with {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.label_type = label_type

        if self.label_type == "classification" or self.label_type == "ppi":
            self.labels_fitness = torch.tensor(
                self.data["label/fitness"].values, dtype=torch.long
            )
        elif self.label_type == "regression":
            self.labels_fitness = torch.tensor(
                self.data["label/fitness"].values, dtype=torch.float32
            )
        elif self.label_type == "multi-label":
            self.data["label/fitness"] = self.data["label/fitness"].apply(
                ast.literal_eval
            )
            self.labels_fitness = torch.tensor(
                self.data["label/fitness"].tolist(), dtype=torch.int32
            )
        else:
            raise ValueError(f"Unsupported label_type: {self.label_type}")

        logger.info(f"Dataset initialized with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label_type == "ppi":
            sequence_1 = self.data.iloc[idx]["sequence_1"]
            sequence_2 = self.data.iloc[idx]["sequence_2"]
            label_fitness = self.labels_fitness[idx]
            return sequence_1, sequence_2, label_fitness
        else:
            sequence = self.data.iloc[idx]["sequence"]
            label_fitness = self.labels_fitness[idx]
            return sequence, label_fitness

class SequenceEmbedding(pl.LightningModule):
    def __init__(
        self, model: pl.LightningModule, tokenizer, is_esm2: bool, output_dir: str
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.is_esm2 = is_esm2
        self.output_dir = output_dir

    def forward(self, sequences: List[str]):
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.is_esm2:
            with torch.no_grad():
                outputs = self.model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            last_hidden_state = outputs.last_hidden_state
            embeddings = last_hidden_state * attention_mask
            embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            return embeddings
        else:
            with torch.no_grad():
                outputs = self.model(inputs["input_ids"])
            return outputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if len(batch) == 2:  # single sequence task
            sequences, labels_fitness = batch
            embeddings = self(sequences)
        elif len(batch) == 3:  # PPI task
            sequences_1, sequences_2, labels_fitness = batch
            embeddings_1 = self(sequences_1)
            embeddings_2 = self(sequences_2)
            embeddings = torch.cat((embeddings_1, embeddings_2), dim=1)
        
        labels_fitness = labels_fitness.to(self.device)

        self.save_embeddings_to_disk(embeddings, labels_fitness, batch_idx)

        return embeddings, labels_fitness

    def save_embeddings_to_disk(self, embeddings, labels_fitness, batch_idx):
        rank = self.global_rank
        output_file = os.path.join(
            self.output_dir, f"embeddings_rank{rank}_batch{batch_idx}.pt"
        )
        torch.save(
            {"embeddings": embeddings.cpu(), "labels_fitness": labels_fitness.cpu()},
            output_file,
        )

def load_custom_model(cfg: DictConfig) -> pl.LightningModule:
    logger.info("Loading custom model configuration")
    
    # Load the model-specific YAML file
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

    model = model.network["sequence"]
    model.eval()
    logger.info("Custom model loaded successfully")
    return model

def combine_embeddings_for_split(split_dir: str, output_file: str):
    logger.info(f"Combining embeddings for split in directory: {split_dir}")
    all_embeddings = []
    all_labels = []

    embedding_files = [
        f
        for f in os.listdir(split_dir)
        if f.startswith("embeddings_rank") and f.endswith(".pt")
    ]

    for file in embedding_files:
        file_path = os.path.join(split_dir, file)
        data = torch.load(file_path)
        all_embeddings.append(data["embeddings"])
        all_labels.append(data["labels_fitness"])

    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    torch.save(
        {"embeddings": final_embeddings, "labels_fitness": final_labels}, output_file
    )

    logger.info(f"Combined embeddings saved to {output_file}")
    logger.info(f"Final embeddings shape: {final_embeddings.shape}")
    logger.info(f"Final labels shape: {final_labels.shape}")

def collate_fn(batch):
    sequences_1, sequences_2, labels = zip(*batch)
    return list(sequences_1), list(sequences_2), torch.stack(labels)

def generate_single_embeddings(
    cfg: DictConfig, task_name: str, task_config: DictConfig, csv_file: str
):
    csv_filename = os.path.basename(csv_file).lower()
    if "train" in csv_filename:
        split = "train"
    elif "valid" in csv_filename:
        split = "valid"
    elif "test" in csv_filename:
        split = "test"
    else:
        raise ValueError(f"Unable to determine split from filename: {csv_file}")

    logger.info(
        f"Generating single embeddings for task: {task_name}, file: {csv_file}, detected split: {split}"
    )

    dataset = SequenceDataset(csv_file, task_config.label_type)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn if task_config.label_type == "ppi" else None,
    )

    if cfg.model.name == "esm2":
        model = AutoModel.from_pretrained(cfg.esm_model)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        is_esm2 = True
    else:
        model = load_custom_model(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        is_esm2 = False

    # Create a unique output directory for each task and split
    task_output_dir = os.path.join(cfg.output_dir, task_name, split, "single_embs")
    os.makedirs(task_output_dir, exist_ok=True)

    sequence_embedding = SequenceEmbedding(model, tokenizer, is_esm2, task_output_dir)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.num_gpus,
        num_nodes=cfg.num_nodes,
        strategy="ddp",
    )

    trainer.predict(sequence_embedding, dataloader)

    logger.info(
        f"Single embeddings for task {task_name} and split {split} saved to {task_output_dir}"
    )

def collect_embeddings(
    cfg: DictConfig, task_name: str, task_config: DictConfig, csv_file: str
):
    csv_filename = os.path.basename(csv_file).lower()
    if "train" in csv_filename:
        split = "train"
    elif "valid" in csv_filename:
        split = "valid"
    elif "test" in csv_filename:
        split = "test"
    else:
        raise ValueError(f"Unable to determine split from filename: {csv_file}")

    logger.info(
        f"Collecting embeddings for task: {task_name}, file: {csv_file}, detected split: {split}"
    )

    single_embs_dir = os.path.join(cfg.output_dir, task_name, split, "single_embs")
    if not os.path.exists(single_embs_dir) or len(os.listdir(single_embs_dir)) == 0:
        logger.info(f"No single embeddings found in {single_embs_dir}. Skipping collection.")
        return

    output_file = os.path.join(
        cfg.output_dir, task_name, split, f"{task_name}_{split}_embeddings_labels.pt"
    )
    combine_embeddings_for_split(single_embs_dir, output_file)

@hydra.main(version_base="1.3", config_path="../configs", config_name="collect_embeddings.yaml")
def main(cfg: DictConfig) -> None:
    logger.info("Starting the embedding generation process")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("CUDA optimizations enabled")

    # Store the original output directory template
    output_dir_template = cfg.output_dir

    for model_config in cfg.models:
        logger.info(f"Processing model: {model_config.name}")
        
        # Create a copy of the config for this model
        model_cfg = OmegaConf.create(cfg)
        
        # Update the model-specific settings
        model_cfg.model = OmegaConf.create(model_config)
        
        # Update the output directory for the current model
        model_cfg.output_dir = output_dir_template.format(model_name=model_config.name)
        
        logger.info(f"Output directory for {model_config.name}: {model_cfg.output_dir}")
        
        for task_name, task_config in model_cfg.tasks.items():
            logger.info(f"Processing task: {task_name}")
            for csv_file in task_config.csv_files:
                if model_cfg.single_batch_mode:
                    generate_single_embeddings(model_cfg, task_name, task_config, csv_file)
                else:
                    collect_embeddings(model_cfg, task_name, task_config, csv_file)

if __name__ == "__main__":
    main()