import os
import logging
from typing import List, Any

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import hydra
from omegaconf import DictConfig
import pyrootutils
import ast
# Set up logging

# Check if this is the first process (SLURM_PROCID == 0)
if int(os.environ.get("SLURM_PROCID", 0)) == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
else:
    # Disable logging for all other processes
    logging.basicConfig(level=logging.CRITICAL)


logger = logging.getLogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class SequenceDataset(Dataset):
    def __init__(self, csv_file: str, label_type: str = "classification"):
        logger.info(f"Initializing SequenceDataset with {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.label_type = label_type

        if self.label_type == "classification":
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
                self.data["label/fitness"], dtype=torch.int32
            )
        else:
            raise ValueError(f"Unsupported label_type: {self.label_type}")

        logger.info(f"Dataset initialized with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
        sequences, labels_fitness = batch
        embeddings = self(sequences)
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
    logger.info("Instantiating custom model")
    model = hydra.utils.instantiate(cfg.model)

    if cfg.ckpt_path is not None:
        logger.info(f"Loading model checkpoint from: {cfg.ckpt_path}")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])
            model.cuda()
        else:
            model.load_state_dict(
                torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
            )

    model = model.oneprot["sequence"]
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

    # Remove individual embedding files and split directory
    for file in embedding_files:
        os.remove(os.path.join(split_dir, file))
    os.rmdir(split_dir)
    logger.info("Individual embedding files and split directory removed")


def process_data(
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
        f"Processing data for task: {task_name}, file: {csv_file}, detected split: {split}"
    )

    dataset = SequenceDataset(csv_file, task_config.label_type)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    if cfg.model_name == "esm2":
        model = AutoModel.from_pretrained(cfg.esm_model)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        is_esm2 = True
    else:
        model = load_custom_model(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        is_esm2 = False

    # Create a unique output directory for each task and split
    task_output_dir = os.path.join(cfg.output_dir, task_name, split)
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
        f"Embeddings for task {task_name} and split {split} saved to {task_output_dir}"
    )
    output_file = os.path.join(
        task_output_dir, f"{task_name}_{split}_embeddings_labels.pt"
    )
    combine_embeddings_for_split(task_output_dir, output_file)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="collect_embeddings.yaml"
)
def main(cfg: DictConfig) -> None:
    logger.info("Starting the embedding generation process")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("CUDA optimizations enabled")

    for task_name, task_config in cfg.tasks.items():
        logger.info(f"Processing task: {task_name}")

        for csv_file in task_config.csv_files:
            process_data(cfg, task_name, task_config, csv_file)


if __name__ == "__main__":
    main()
