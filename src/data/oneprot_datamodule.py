from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from omegaconf import DictConfig
import logging

from src.data.datasets.msa_dataset import MSADataset
from src.data.datasets.struct_graph_dataset import StructDataset
from src.data.datasets.text_dataset import TextDataset
from src.data.datasets.struct_token_dataset import StructTokenDataset
from src.data.datasets.seqsim_dataset import SequenceSimDataset

# Dictionary mapping modality names to their respective dataset classes
DATASET_CLASSES = {
    "msa": MSADataset,
    "struct_graph": StructDataset,
    "pocket": StructDataset,
    "text": TextDataset,
    "struct_token": StructTokenDataset,
    "seqsim": SequenceSimDataset,

}

class OneProtDataModule(LightningDataModule):
    def __init__(self, modalities: DictConfig, num_workers, pin_memory, default_batch_size):
        super().__init__()
        self.modalities = modalities
        self.datasets: Dict[str, Any] = {}
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.default_batch_size = default_batch_size
        self.logger = logging.getLogger(__name__)

    def setup(self, stage: Optional[str] = None):
        
        if not self.datasets:
            for modality, modality_cfg in self.modalities.items():
                if modality not in DATASET_CLASSES:
                    self.logger.error(f"Unknown modality: {modality}")
                    continue
                
                dataset_class = DATASET_CLASSES[modality]
                for split in ['train', 'val', 'test']:
                    dataset_kwargs = {**modality_cfg.dataset, 'split': split}
                    dataset_kwargs.pop('_target_', None)  # Remove _target_ if it exists
                    try:
                        self.datasets[f"{modality}_{split}"] = dataset_class(**dataset_kwargs)
                    except Exception as e:
                        self.logger.error(f"Error creating dataset for {modality} {split}: {str(e)}")
                        continue
                self.logger.info(f"{modality} Train/Validation/Test Dataset Size = "
                                 f"{len(self.datasets.get(f'{modality}_train', []))} / "
                                 f"{len(self.datasets.get(f'{modality}_val', []))} / "
                                 f"{len(self.datasets.get(f'{modality}_test', []))}")

    def _create_dataloader(self, split: str, shuffle: bool = False):
        iterables = {}
        for modality, modality_cfg in self.modalities.items():
            if f"{modality}_{split}" not in self.datasets:
                self.logger.warning(f"Dataset for {modality} {split} not found, skipping.")
                continue
            batch_size = modality_cfg.batch_size.get(split, self.default_batch_size)

            iterables[modality] = DataLoader(
                dataset=self.datasets[f"{modality}_{split}"],
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.datasets[f"{modality}_{split}"].collate_fn,
                shuffle=shuffle,
                drop_last=False,
            )

        return CombinedLoader(iterables, "min_size_cycle" if shuffle else "sequential")

    def train_dataloader(self):
        return self._create_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader("val")

    def test_dataloader(self):
        return self._create_dataloader("test")