from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import os
from src.data.components.datasets import MSADataset, StructDataset, PocketDataset, TextDataset

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
        data_dir: str = "/p/scratch/hai_oneprot/Dataset_25_06_24",
        data_modalities: list = ['sequence','structure','pocket'],
        text_tokenizer: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        seq_tokenizer: str = "facebook/esm2_t12_35M_UR50D",
        use_struct_mask: bool = False, 
        use_struct_coord_noise: bool = False, 
        use_struct_deform: bool =False,
        batch_size: int = 64,
        pin_memory: bool = False,
        pocket_data_type='h5',
        seqsim='30ss'
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        
        self.num_workers =  int(os.getenv('SLURM_CPUS_PER_TASK'))
        self.save_hyperparameters(logger=False)
        self.data_modalities = data_modalities
        self.data_dir = data_dir
        self.seq_tokenizer = seq_tokenizer
        self.text_tokenizer = text_tokenizer
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.pocket_data_type = pocket_data_type
        self.seqsim=seqsim


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            self.datasets = {}
            self.datasets_collate_fn = {}
            for modality in self.data_modalities:
                
                if modality == 'struct':
                    #print(modality," modality")
                    self.datasets["struct_train"] =  StructDataset(data_dir =self.data_dir, split='train', seq_tokenizer=self.seq_tokenizer, use_struct_mask=self.hparams.use_struct_mask, use_struct_coord_noise=self.hparams.use_struct_coord_noise, use_struct_deform=self.hparams.use_struct_deform,seqsim=self.seqsim )
                    self.datasets["struct_val"] =  StructDataset(data_dir =self.data_dir, split='val', seq_tokenizer=self.seq_tokenizer,seqsim=self.seqsim)
                    self.datasets["struct_test"] =  StructDataset(data_dir =self.data_dir, split='test', seq_tokenizer=self.seq_tokenizer,seqsim=self.seqsim)
                  
                    
                elif modality == 'msa':
                    #print(modality," modality")
                    self.datasets["msa_train"] =  MSADataset(data_dir =self.data_dir, split='train', seq_tokenizer=self.seq_tokenizer,seqsim=self.seqsim)
                    self.datasets["msa_val"] =  MSADataset(data_dir =self.data_dir, split='val', seq_tokenizer=self.seq_tokenizer,seqsim=self.seqsim)
                    self.datasets["msa_test"] =  MSADataset(data_dir =self.data_dir, split='test', seq_tokenizer=self.seq_tokenizer,seqsim=self.seqsim)
                
                elif modality == 'pocket':
                    #print(modality," modality")
                    # self.datasets["pocket_train"] =  PocketDataset(split='train', seq_tokenizer=self.seq_tokenizer,data_type=self.pocket_data_type)
                    # self.datasets["pocket_val"] =  PocketDataset(split='val', seq_tokenizer=self.seq_tokenizer,data_type=self.pocket_data_type)
                    # self.datasets["pocket_test"] =  PocketDataset(split='test', seq_tokenizer=self.seq_tokenizer, data_type=self.pocket_data_type)
                    self.datasets["pocket_train"] =  StructDataset(data_dir =self.data_dir, split='train', seq_tokenizer=self.seq_tokenizer, use_struct_mask=self.hparams.use_struct_mask, use_struct_coord_noise=self.hparams.use_struct_coord_noise, use_struct_deform=self.hparams.use_struct_deform,pockets=True,seqsim=self.seqsim )
                    self.datasets["pocket_val"] =  StructDataset(data_dir =self.data_dir, split='val', seq_tokenizer=self.seq_tokenizer,pockets=True,seqsim=self.seqsim)
                    self.datasets["pocket_test"] =  StructDataset(data_dir =self.data_dir, split='test', seq_tokenizer=self.seq_tokenizer,pockets=True,seqsim=self.seqsim)
              

                
                elif modality == 'text':
                    #print(modality," modality")
                    self.datasets["text_train"] =  TextDataset(data_dir =self.data_dir, split='train', seq_tokenizer=self.seq_tokenizer, text_tokenizer=self.text_tokenizer,seqsim=self.seqsim)
                    self.datasets["text_val"] =  TextDataset(data_dir =self.data_dir, split='val', seq_tokenizer=self.seq_tokenizer, text_tokenizer=self.text_tokenizer,seqsim=self.seqsim)
                    self.datasets["text_test"] =  TextDataset(data_dir =self.data_dir, split='test', seq_tokenizer=self.seq_tokenizer, text_tokenizer=self.text_tokenizer,seqsim=self.seqsim)
                
                print(f"{modality} Train/Validation/Test Dataset Size = {len(self.datasets[f'{modality}_train'])} / {len(self.datasets[f'{modality}_val'])} / {len(self.datasets[f'{modality}_test'])}")
                
    def train_dataloader(self):
        iterables = {}
        for modality in self.data_modalities:
        
            iterables[modality] = DataLoader(
                        dataset=self.datasets[f"{modality}_train"],
                        batch_size=self.hparams.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        collate_fn=self.datasets[f"{modality}_train"].collate_fn,
                        shuffle=True,
                        drop_last=True,
                    )

        return CombinedLoader(iterables, 'min_size')

    def val_dataloader(self):
        iterables = {}
        for modality in self.data_modalities:
            iterables[modality] = DataLoader(
                        dataset=self.datasets[f"{modality}_val"],
                        batch_size=self.hparams.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        collate_fn=self.datasets[f"{modality}_val"].collate_fn,
                        drop_last=True,
                        shuffle=False,
                    )


        return CombinedLoader(iterables, 'sequential')

    def test_dataloader(self):
        iterables = {}
        for modality in self.data_modalities:
        
            iterables[modality] = DataLoader(
                        dataset=self.datasets[f"{modality}_test"],
                        batch_size=self.hparams.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        collate_fn=self.datasets[f"{modality}_test"].collate_fn,
                        drop_last=True,
                        shuffle=False,
                    )

        return CombinedLoader(iterables, 'sequential')
        
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
