from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader



from src.data.components.datasets import MSADataset, GODataset, TextDataset, StructureDataset
from src.data.components.datasets import structure_collate_fn


class ONEPROTDataModule(LightningDataModule):
    """Example of LightningDataModule for ONEPROT dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        data_modalities: list = ['sequence','structure'],
        train_val_test_split: Tuple[float, float, float] = (0.9, 0.05, 0.05),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_modalities = data_modalities
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            self.datasets = {}
            for modality in self.data_modalities:
                if modality == 'go':
                    dataset = GODataset()
                elif modality == 'structure':
                    dataset = StructureDataset()
                elif modality == 'text':
                    dataset = TextDataset()
                elif modality == 'msa':
                    dataset = MSADataset()
                
                print(f" {modality} Dataset Size = {len(dataset)}")

                train_len = int(self.hparams.train_val_test_split[0]*(len(dataset)))
                val_len = int(self.hparams.train_val_test_split[1]*(len(dataset)))
                test_len = (len(dataset) - train_len - val_len)
                
                data_train, data_val, data_test = random_split(
                    dataset=dataset,
                    lengths=[train_len, val_len, test_len],
                    generator=torch.Generator().manual_seed(42),
                )
                self.datasets[f"{modality}_train"] = data_train
                self.datasets[f"{modality}_val"] = data_val
                self.datasets[f"{modality}_test"] = data_test
            
    def train_dataloader(self):
        
        if len(self.data_modalities)>1:
            iterables = {}
            for modality in self.data_modalities:
            
                if modality != 'structure' :
                    iterables[modality] = DataLoader(
                                dataset=self.datasets[f"{modality}_train"],
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                pin_memory=self.hparams.pin_memory,
                                shuffle=True,
                            )
                else:
                    iterables[modality] = DataLoader(
                            dataset=self.datasets["structure_train"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            collate_fn=structure_collate_fn,
                            shuffle=True,
                        )
            return CombinedLoader(iterables, 'min_size')
        else:
      
            if 'structure' not in self.data_modalities:
                return DataLoader(
                            dataset=self.datasets[f"{self.data_modalities[0]}_train"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=True,
                        )
            
            else:
                return DataLoader(
                            dataset=self.datasets["structure_train"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            collate_fn=structure_collate_fn,
                            shuffle=True,
                        )

    def val_dataloader(self):
        
        if len(self.data_modalities)>1:
            iterables = {}
            for modality in self.data_modalities:
            
                if modality != 'structure':
                    iterables[modality] = DataLoader(
                                dataset=self.datasets[f"{modality}_val"],
                                batch_size=4*self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                pin_memory=self.hparams.pin_memory,
                                shuffle=False,
                            )
                else:
                    iterables[modality] = DataLoader(
                            dataset=self.datasets["structure_val"],
                            batch_size=4*self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            collate_fn=structure_collate_fn,
                            shuffle=False,
                        )
            return CombinedLoader(iterables, 'sequential')
        else:

            if 'structure' not in self.data_modalities:
                return DataLoader(
                            dataset=self.datasets[f"{modality}_val"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=False,
                        )
            
            else:
                return DataLoader(
                            dataset=self.datasets["structure_val"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            collate_fn=structure_collate_fn,
                            shuffle=False,
                        )
            
    def test_dataloader(self):
        
        if len(self.data_modalities)>1:
            iterables = {}
            for modality in self.data_modalities:
            
                if modality != 'structure' :
                    iterables[modality] = DataLoader(
                                dataset=self.datasets[f"{self.data_modalities[0]}_test"],
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                pin_memory=self.hparams.pin_memory,
                                shuffle=False,
                            )
                else:
                    iterables[modality] = DataLoader(
                            dataset=self.datasets["structure_test"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            collate_fn=structure_collate_fn,
                            shuffle=False,
                        )
            return CombinedLoader(iterables, 'sequential')
        else:

            if modality != 'structure' :
                return DataLoader(
                            dataset=self.datasets[f"{self.data_modalities[0]}_test"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=False,
                        )
            
            else:
                return DataLoader(
                            dataset=self.datasets["structure_test"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            collate_fn=structure_collate_fn,
                            shuffle=False,
                        )
        
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
