from typing import Any, Dict, Tuple
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from src.models.components.loss import ClipLoss, SigLipLoss
from src.models.components.retrieval_metric import RetrievalMetric
import os
from collections import defaultdict
class OneProtLitModule(LightningModule):
    def __init__(
        self,
        components: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        use_seqsim: bool = False,
        loss_fn: str = 'CLIP',
        use_l1_regularization: bool = False,
        local_loss: bool = True,
        gather_with_grad: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(logger=False)
        
        self.network = torch.nn.ModuleDict(components)
        self.modalities = list(components.keys())
        self.use_l1_regularization = use_l1_regularization
        self.loss_fn = self._create_loss_fn(loss_fn, local_loss, gather_with_grad)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.use_seqsim = use_seqsim
        

        self.metrics = {
            f"{split}_{modality}": RetrievalMetric()
            for split in ["val", "test"]
            for modality in list(self.network.keys()) + ["seqsim"]
            if modality != 'sequence'
        }

    def l1_regularization(self, features):
            return torch.abs(features).mean()

    def _create_loss_fn(self, loss_fn, local_loss, gather_with_grad):
        if loss_fn == 'CLIP':
            return ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=True,
                rank=int(os.environ['RANK']),
                world_size=int(os.environ['WORLD_SIZE']),
            )
        elif loss_fn == 'SIGLIP':
            return SigLipLoss(
                cache_labels=True,
                rank=int(os.environ['RANK']),
                world_size=int(os.environ['WORLD_SIZE']),
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")


    def forward(self, x, modality="sequence"):

        if modality in ["sequence", "seqsim"]:
            modality = "sequence"
        return self.network[modality](x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_loss_best.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:    
        
        opt = self.optimizers()    
        
        for modality, inputs in list(batch.items()):     
            if not self.use_seqsim and modality =="seqsim":
                continue
            
            sequence_inputs, modality_inputs, modality, _ = inputs
            sequence_features = self.forward(sequence_inputs, "sequence")
            modality_features = self.forward(modality_inputs, modality)
            opt.zero_grad()
            
            if self.use_l1_regularization:
                loss = self.loss_fn(sequence_features, modality_features)
                loss +=  0.001 * (torch.abs(sequence_features).mean() + torch.abs(modality_features).mean())
            else:
                loss = self.loss_fn(sequence_features, modality_features)
            self.train_loss(loss)
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
 
    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        sequence_inputs, modality_inputs, modality, _ = batch
 
        sequence_features = self.forward(sequence_inputs, "sequence")
        modality_features = self.forward(modality_inputs, modality)
        
        self.metrics["val_"+modality].update(sequence_features, modality_features)

        loss = self.loss_fn(sequence_features, modality_features)
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

    def on_validation_epoch_end(self):
                
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        
        for modality in self.metrics:
            if modality.startswith("val_"):
                metric_results = self.metrics[modality].compute()
                for key, value in metric_results.items():
                    self.log(f"val/{key}/{modality}", value, sync_dist=True, prog_bar=True)
                self.metrics[modality].reset()    
    
    def test_step(self, batch, batch_idx):
        for modality, (seq_inputs, mod_inputs, _, _) in batch.items():
            seq_features = self(seq_inputs, "sequence")
            mod_features = self(mod_inputs, modality)
            
            loss = self.loss_fn(seq_features, mod_features, self.network[modality].norm[1].log_logit_scale.exp())
            self.test_loss(loss)
            self.log(f"test/loss_{modality}", loss, on_step=False, on_epoch=True, prog_bar=True)
            
            self.metrics[f"test_{modality}"].update(seq_features, mod_features)

    def on_test_epoch_end(self):
        for modality in self.metrics:
            if modality.startswith("test_"):
                metric_results = self.metrics[modality].compute()
                for key, value in metric_results.items():
                    self.log(f"test/{key}/{modality}", value, prog_bar=True)
                self.metrics[modality].reset()


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}