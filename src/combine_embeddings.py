import os
import logging
import torch
import hydra
from omegaconf import DictConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_embeddings_for_split(split_dir: str, output_file: str):
    logger.info(f"Combining embeddings for split in directory: {split_dir}")
    all_embeddings = []
    all_labels = []

    embedding_files = [f for f in os.listdir(split_dir) if f.startswith('embeddings_rank') and f.endswith('.pt')]

    for file in embedding_files:
        file_path = os.path.join(split_dir, file)
        data = torch.load(file_path)
        all_embeddings.append(data['embeddings'])
        all_labels.append(data['labels_fitness'])

    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    torch.save({
        'embeddings': final_embeddings,
        'labels_fitness': final_labels
    }, output_file)

    logger.info(f"Combined embeddings saved to {output_file}")
    logger.info(f"Final embeddings shape: {final_embeddings.shape}")
    logger.info(f"Final labels shape: {final_labels.shape}")

    # Remove individual embedding files
    for file in embedding_files:
        os.remove(os.path.join(split_dir, file))
    logger.info("Individual embedding files removed")

@hydra.main(version_base="1.3", config_path="../configs", config_name="esm_saprot.yaml")
def main(cfg: DictConfig) -> None:
    logger.info("Starting the embedding combination process")

    for task_name in cfg.tasks.keys():
        task_dir = os.path.join(cfg.output_dir, task_name)
        if not os.path.exists(task_dir):
            logger.warning(f"Task directory not found: {task_dir}")
            continue

        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(task_dir, split)
            if not os.path.exists(split_dir):
                logger.info(f"Split directory not found: {split_dir}. Skipping.")
                continue

            output_file = os.path.join(cfg.output_dir, f"{task_name}_{split}_combined_embeddings.pt")
            combine_embeddings_for_split(split_dir, output_file)

    logger.info("Embedding combination process completed")

if __name__ == "__main__":
    main()