from typing import Any, Dict, Tuple
import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torch import nn
#from pytorch_metric_learning import losses
from src.models.components.loss import ClipLoss, SigLipLoss
from src.models.components.retrieval_metric import RetrievalMetric
from collections import Counter
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


class ONEPROTLitModule(LightningModule):

    def __init__(
        self,
        network: torch.nn.ModuleDict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        loss_fn: str = 'CLIP',
        local_loss: bool = True,
        gather_with_grad: bool = True,
        compile: bool = False,

    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        self.automatic_optimization = False
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.oneprot = network
        
        self.data_modalities = [key for key in self.oneprot.keys() if key != 'sequence']
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
     
        if loss_fn == 'CLIP':
            
            self.loss_fn = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=True,
                rank=int(os.environ['RANK']),
                world_size=int(os.environ['WORLD_SIZE']),
            )
        
        elif loss_fn == 'SIGLIP':
            self.loss_fn = SigLipLoss(
                    cache_labels=True,
                    rank=int(os.environ['RANK']),
                    world_size=int(os.environ['WORLD_SIZE']),
                )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()
        self.metrics = {}
        for modality in self.data_modalities:

            self.metrics["val_"+modality] = RetrievalMetric()
            self.metrics["test_"+modality] = RetrievalMetric()

    def forward(self, sequence_inputs, modality_inputs, modality = 'structure') -> torch.Tensor:
        #print("I am in the forward of the model!!!!!!!!!!!!!!!!!!")
        sequence_outputs = self.oneprot['sequence'](sequence_inputs)

        if modality=='msa':
            modality_output_temp = []
            for i in range(0, modality_inputs.shape[0], 4):
                #modality_output = self.oneprot[modality](torch.unsqueeze(modality_inputs[i,...],0))
                #print(modality_inputs[i:i+4,...].shape," modality inputs!!!!!!!!!!!!!!!!")
                modality_output = self.oneprot[modality](modality_inputs[i:i+4,...])
                modality_output_temp.extend(modality_output)
            modality_outputs = torch.stack(modality_output_temp)
        else:
            modality_outputs = self.oneprot[modality](modality_inputs)
        return sequence_outputs, modality_outputs

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], #batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        opt = self.optimizers()    
        
        for modality in list(batch.keys()):     
            sequence_inputs, modality_inputs = batch[modality]
            sequence_features, modality_features = self.forward(sequence_inputs, modality_inputs, modality)
            opt.zero_grad()
            loss = self.loss_fn(sequence_features, modality_features)
            self.train_loss(loss)
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()
            self.log(f"train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor] = None, batch_idx: int = 0, 
                        dataloader_idx: int = 0) -> None:

        modality = self.data_modalities[dataloader_idx]
        sequence_inputs, modality_inputs = batch
        sequence_features, modality_features = self.forward(sequence_inputs, modality_inputs, modality)
        self.metrics["val_"+modality].update(sequence_features, modality_features)

        loss = self.loss_fn(sequence_features, modality_features)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        for modality in self.data_modalities:
            metric_results = self.metrics["val_"+modality].compute()

            for vals in metric_results:
                self.log(f"val/{modality}/{vals}", metric_results[vals], sync_dist=True, prog_bar=True)
            self.metrics["val_"+modality].reset()
        
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor] = None, batch_idx: int = 0, 
                  dataloader_idx: int = 0) -> None:

        
        modality = self.data_modalities[dataloader_idx]
        sequence_inputs, modality_inputs = batch
        sequence_features, modality_features = self.forward(sequence_inputs, modality_inputs, modality)
     
        self.metrics["test_"+modality].update(sequence_features, modality_features)

        loss = self.loss_fn(sequence_features, modality_features)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

    def on_test_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.test_loss.compute()  # get current val acc
        
        for modality in self.data_modalities:

            metric_results = self.metrics["test_"+modality].compute()
            for vals in metric_results:
                self.log(f"test/{modality}/{vals}", metric_results[vals], sync_dist=True, prog_bar=True)
            self.metrics["test_"+modality].reset()
        
        self.log("test/loss", loss, sync_dist=True, prog_bar=True)
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
       
        return optimizer
        
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.oneprot = torch.compile(self.oneprot)