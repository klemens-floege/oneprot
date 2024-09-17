from typing import List, Optional, Dict, Any
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import os

from msa_dataset import MSADataset
from struct_graph_dataset import StructDataset
from text_dataset import TextDataset
from struct_token_dataset import StructTokenDataset

class ONEPROTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/p/scratch/hai_oneprot/Dataset_25_06_24",
        data_modalities: List[str] = ["msa", "struct_graph", "text", "struct_token", "pocket"],
        text_tokenizer: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D",
        struct_tokenizer: str = "westlake-repl/SaProt_650M_AF2",
        use_struct_mask: bool = False,
        use_struct_coord_noise: bool = False,
        use_struct_deform: bool = False,
        batch_size: int = 64,
        pin_memory: bool = False,
        seqsim: str = "30ss",
        msa_depth: int = 100,
        max_length: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_dir = data_dir
        self.data_modalities = data_modalities
        self.seq_tokenizer = seq_tokenizer
        self.text_tokenizer = text_tokenizer
        self.struct_tokenizer = struct_tokenizer
        self.seqsim = seqsim
        
        self.num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        self.datasets: Dict[str, Any] = {}

    def setup(self, stage: Optional[str] = None):
        if not self.datasets:
            for modality in self.data_modalities:
                for split in ['train', 'val', 'test']:
                    dataset_class = self._get_dataset_class(modality)
                    dataset_kwargs = self._get_dataset_kwargs(modality, split)
                    self.datasets[f"{modality}_{split}"] = dataset_class(**dataset_kwargs)
                print(f"{modality} Train/Validation/Test Dataset Size = "
                      f"{len(self.datasets[f'{modality}_train'])} / "
                      f"{len(self.datasets[f'{modality}_val'])} / "
                      f"{len(self.datasets[f'{modality}_test'])}")

    def _get_dataset_class(self, modality: str):
        if modality == "msa":
            return MSADataset
        elif modality == "struct_graph":
            return StructDataset
        elif modality == "pocket":
            return StructDataset
        elif modality == "text":
            return TextDataset
        elif modality == "struct_token":
            return StructTokenDataset
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _get_dataset_kwargs(self, modality: str, split: str) -> Dict[str, Any]:
        common_kwargs = {
            "data_dir": self.data_dir,
            "split": split,
            "seqsim": self.hparams.seqsim,
            "seq_tokenizer": self.hparams.seq_tokenizer,
        }
        
        if modality == "msa":
            return {
                **common_kwargs,
                "max_length": self.hparams.max_length,
                "msa_depth": self.hparams.msa_depth,
              
            }
        elif modality == "struct_graph":
            return {
                **common_kwargs,
                "use_struct_mask": self.hparams.use_struct_mask,
                "use_struct_coord_noise": self.hparams.use_struct_coord_noise,
                "use_struct_deform": self.hparams.use_struct_deform,
                "pockets": False,
            }
        elif modality == "pocket":
            return {
                **common_kwargs,
                "use_struct_mask": self.hparams.use_struct_mask,
                "use_struct_coord_noise": self.hparams.use_struct_coord_noise,
                "use_struct_deform": self.hparams.use_struct_deform,
                "pockets": True,
            }
        elif modality == "text":
            return {
                **common_kwargs,
                "text_tokenizer": self.hparams.text_tokenizer,
            }
        elif modality == "struct_token":
            return {
                **common_kwargs,
                "struct_tokenizer": self.hparams.struct_tokenizer,
            }
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _create_dataloader(self, split: str, shuffle: bool = False):
        iterables = {}
        for modality in self.data_modalities:
            iterables[modality] = DataLoader(
                dataset=self.datasets[f"{modality}_{split}"],
                batch_size=self.hparams.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.datasets[f"{modality}_{split}"].collate_fn,
                shuffle=shuffle,
                drop_last=True,
            )
        return CombinedLoader(iterables, "min_size" if shuffle else "sequential")

    def train_dataloader(self):
        return self._create_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader("val")

    def test_dataloader(self):
        return self._create_dataloader("test")

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass