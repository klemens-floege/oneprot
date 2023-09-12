from typing import Any, Dict, Tuple
import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torch import nn, einsum
from torch.optim import Adam, AdamW
#from pytorch_metric_learning import losses
from src.models.components.loss import ClipLoss
from src.models.components.sequence_model import SequenceModel
from src.models.components.struct_model import StructModel
from src.models.components.text_model import TextModel
from src.models.components.go_model import GoModel
from src.models.components.msa_model import MsaModel

from collections import Counter
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class ONEPROTLitModule(LightningModule):


    def __init__(
        self,
        data_modalities: list = ['text','structure'],
        output_dim: int = 1024, 
        sequence_model: str =  "facebook/esm2_t12_35M_UR50D", #"facebook/esm2_t12_35M_UR50D",
        text_model: str =  "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",       
        struct_proj = 'mlp',
        sequence_proj = 'mlp',
        go_proj= 'mlp',
        text_proj= 'mlp',
        msa_proj= 'mlp',
        use_logit_scale: bool = True,
        local_loss: bool = True,
        gather_with_grad: bool = True,
        struct_level: str = 'allatom',
        mask_aatype: float = 0.2,
        struct_mask: bool = True,
        struct_noise: bool = True,
        struct_deform: bool = True,
        lr: float = 5e-4,
        weight_decay: float = 1e-4, 
        max_epochs:int = 100
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_modalities = data_modalities
        self.automatic_optimization = False

        self.validation_step_outputs = {}

        oneprot_model = {}
        oneprot_model['sequence'] = SequenceModel(model_name_or_path=sequence_model, output_dim=output_dim, proj=sequence_proj, use_logit_scale=use_logit_scale)
        for modality in self.data_modalities:
            if modality == 'msa':
                oneprot_model[modality] = MsaModel(output_dim=output_dim, proj=msa_proj, use_logit_scale=use_logit_scale)
            elif modality == 'structure':
                oneprot_model[modality] = StructModel(output_dim=output_dim, proj=struct_proj, use_logit_scale=use_logit_scale)
            elif modality == 'text':
                oneprot_model[modality] = TextModel(text_model,output_dim=output_dim, proj=text_proj, use_logit_scale=use_logit_scale)
            elif modality == 'go':
                oneprot_model[modality] = GoModel(output_dim=output_dim, proj=go_proj, use_logit_scale=use_logit_scale)

        self.oneprot_module = nn.ModuleDict(oneprot_model)

        self.loss_fn = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=True,
                rank=int(os.environ['RANK']),
                world_size=int(os.environ['WORLD_SIZE']),
            )
        
        self.struct_mask = struct_mask
        self.struct_noise = struct_noise
        self.struct_deform = struct_deform
        self.mask_aatype = mask_aatype
        self.struct_level = struct_level
        self.all_sequence_features={}
        for modality in self.data_modalities:
            self.all_sequence_features[modality] = []
        
        self.all_modality_features={}
        for modality in self.data_modalities:
            self.all_modality_features[modality] = []
 
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, batch) -> torch.Tensor:
 
        modality_outputs = {} 
        sequence_outputs = {}
        for modality in list(batch.keys()):
            
            
            sequence_features, modality_features = batch[modality]

            sequence_output = self.oneprot_module['sequence'](sequence_features)
            #sequence_outputs.extend(sequence_output)
            sequence_outputs[modality] = sequence_output
            
            if modality=='msa':
                modality_output_temp = []
                for i in range(modality_features.shape[0]):
                    modality_output = self.oneprot_module[modality](torch.unsqueeze(modality_features[i,...],0))
                    modality_output_temp.extend(modality_output)
                modality_outputs[modality] = torch.stack(modality_output_temp)
            else:
                modality_output = self.oneprot_module[modality](modality_features)
                #modality_outputs.extend(modality_output)
                modality_outputs[modality] = modality_output
        return sequence_outputs, modality_outputs

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        opt = self.optimizers()
        
        for modality in list(batch.keys()):     
            temp_batch = {}
            

            if modality == 'structure':
                sequence_features, modality_features = batch[modality]
                if self.struct_mask:
                    # random mask node aatype
                    mask_indice = torch.tensor(np.random.choice(modality_features.num_nodes, int(modality_features.num_nodes * self.mask_aatype), replace=False))
                    modality_features.x[:, 0][mask_indice] = 25
                if self.struct_noise:
                    # add gaussian noise to atom coords
                    gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=modality_features.coords_ca.shape), min=-0.3, max=0.3).to(self.device)
                    modality_features.coords_ca += gaussian_noise
                    if self.struct_level != 'aminoacid':
                        modality_features.coords_n += gaussian_noise
                        modality_features.coords_c += gaussian_noise
                if self.struct_deform:
                    # Anisotropic scale
                    deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1).to(self.device)
                    modality_features.coords_ca *= deform
                    if self.struct_level != 'aminoacid':
                        modality_features.coords_n *= deform
                        modality_features.coords_c *= deform
                temp_batch[modality] = (sequence_features, modality_features)
            else:
                temp_batch[modality] = batch[modality]
            sequence_outputs, modality_outputs = self.forward(temp_batch)
            opt.zero_grad()
            loss = self.loss_fn(sequence_outputs[modality], modality_outputs[modality])
            self.train_loss(loss)
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
            #print("passed first")
            
            metrics = self.get_clip_metrics(sequence_outputs[modality], modality_outputs[modality])
        
            self.log(f"train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            
            for name, value in metrics.items():
                self.log(f'train/{modality}/{name}', value,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        #return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor] = None, batch_idx: int = 0, dataloader_idx: int = 0) -> None:

        batch_data = {}
        batch_data[self.data_modalities[dataloader_idx]]= batch
        sequence_outputs, modality_outputs = self.forward(batch_data)

        #self.all_modality_features[modality]

        if self.data_modalities[dataloader_idx] in self.validation_step_outputs:
            self.validation_step_outputs[self.data_modalities[dataloader_idx]]['seq'].extend(sequence_outputs[self.data_modalities[dataloader_idx]])
            self.validation_step_outputs[self.data_modalities[dataloader_idx]]['mod'].extend(modality_outputs[self.data_modalities[dataloader_idx]])
        else:
            temp_seq = []
            temp_mod = []
            temp_seq.extend(sequence_outputs[self.data_modalities[dataloader_idx]])
            temp_mod.extend(modality_outputs[self.data_modalities[dataloader_idx]])
            self.validation_step_outputs[self.data_modalities[dataloader_idx]] = {}
            self.validation_step_outputs[self.data_modalities[dataloader_idx]]['seq'] = temp_seq
            self.validation_step_outputs[self.data_modalities[dataloader_idx]]['mod'] = temp_mod
        #print(f"modality {self.data_modalities[dataloader_idx]}")
        loss = self.loss_fn(sequence_outputs[self.data_modalities[dataloader_idx]], modality_outputs[self.data_modalities[dataloader_idx]])
        
        #metrics = self.get_clip_metrics(sequence_outputs, modality_outputs)
        
        self.val_loss(loss)
        #self.log('train_loss_ml', nll)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #for name, value in metrics.items():
        #    self.log(f'val/{name}', value,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        
        for modality in self.data_modalities:
        
            metrics = self.get_clip_metrics(torch.stack(self.validation_step_outputs[modality]['seq']).cpu(), torch.stack(self.validation_step_outputs[modality]['mod']).cpu())
            
            for name, value in metrics.items():
                self.log(f'val/{modality}/{name}', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:

        total_loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(total_loss)
  
        # return loss or backpropagation will fail    
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
       
        return optimizer
    

    def get_clip_metrics(self, sequence_outputs, modality_outputs):
            metrics = {}
            logits_per_sequence = (sequence_outputs @ modality_outputs.t()).detach().cpu()
            logits_per_modality = logits_per_sequence.t().detach().cpu()

            logits = {"seq_to_mod": logits_per_sequence, "mod_to_seq": logits_per_modality}
            ground_truth = torch.arange(len(modality_outputs)).view(-1, 1)

            for name, logit in logits.items():
                ranking = torch.argsort(logit, descending=True)
                preds = torch.where(ranking == ground_truth)[1]
                preds = preds.detach().cpu().numpy()
                metrics[f"{name}_mean_rank"] = preds.mean() + 1
                metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
                for k in [1, 5, 10]:
                    metrics[f"{name}_R@{k}"] = np.mean(preds < k)

            return metrics
