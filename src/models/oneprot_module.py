from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
import esm
from transformers import BertConfig, AutoTokenizer, EsmModel, AutoModel
from torch import nn, einsum
from torch.optim import Adam, AdamW
from pytorch_metric_learning import losses
from src.models.components.featpronet import FeatProNet
from src.models.components.simpletransformer import GOBertEmbeddings, GOBertModel, TrunkBertModel, ProtNETBertModel, SelectElement, LearnableLogitScaling, Normalize
from collections import Counter
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch.nn.functional as F
import torch.optim as optim

class ONEPROTLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        data_modalities: list = ['text','structure'],
        out_embed_dim: int = 1024, 
        sequence_model: str =  "facebook/esm2_t30_150M_UR50D", #"facebook/esm2_t12_35M_UR50D",
        sequence_hidden_size: int = 480,
        msa_hidden_size: int = 768,
        text_hidden_size: int = 768,
        structure_hidden_size: int = 768,
        go_hidden_size: int = 768,
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
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.data_modalities = data_modalities
        oneprot_model = {}
        sequence_model = EsmModel.from_pretrained(sequence_model)
        sequence_model.eval()
        for param in sequence_model.parameters():
                param.requires_grad = False
        oneprot_model["sequence_model"] = sequence_model
        oneprot_model["sequence_head"] = nn.Sequential(
                    nn.LayerNorm(normalized_shape=sequence_hidden_size, eps=1e-6),
                    SelectElement(index=0),
                    nn.Dropout(p=0.5),
                    nn.Linear(sequence_hidden_size, out_embed_dim, bias=False),
                )
        oneprot_model["sequence_post"] = nn.Sequential(
                    Normalize(dim=-1), LearnableLogitScaling(learnable=True)
                )

        for modality in self.data_modalities:
        
            if modality=='msa':
                msa_model, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()  
                msa_model.eval() 
                for param in msa_model.parameters():
                    param.requires_grad = False
                        
                oneprot_model["msa_model"] = msa_model
                configuration = BertConfig(hidden_size=msa_hidden_size, max_position_embeddings=1024)
                oneprot_model['msa_head'] = nn.Sequential(
                            nn.LayerNorm(normalized_shape=msa_hidden_size, eps=1e-6),
                            SelectElement(index=0),
                            nn.Dropout(p=0.5),
                            nn.Linear(msa_hidden_size, out_embed_dim, bias=False),
                        )
                oneprot_model['msa_post'] = nn.Sequential(
                            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
                        )

            elif modality=='structure':
                #self.aa_protein_emb = FeatProNet(level='aminoacid', hidden_channels=structure_hidden_size)
                #self.backbone_protein_emb = FeatProNet(level='backbone', hidden_channels=structure_hidden_size)
                oneprot_model["structure_model"] = FeatProNet(level='allatom', hidden_channels=structure_hidden_size, out_channels = structure_hidden_size)
                oneprot_model['structure_head'] = nn.Sequential(
                            #nn.LayerNorm(normalized_shape=structure_hidden_size, eps=1e-6),
                            #nn.Dropout(p=0.5),
                            nn.Linear(structure_hidden_size, out_embed_dim, bias=False),
                        )
                oneprot_model["structure_post"] = nn.Sequential(
                            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
                        )

            elif modality=='text':

                text_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
                text_model.eval()
                for param in text_model.parameters():
                        param.requires_grad = False
                oneprot_model["text_model"] = text_model
                oneprot_model["text_head"] = nn.Sequential(
                            nn.LayerNorm(normalized_shape=text_hidden_size, eps=1e-6),
                            SelectElement(index=0),
                            nn.Dropout(p=0.5),
                            nn.Linear(text_hidden_size, out_embed_dim, bias=False),
                        )
                oneprot_model["text_post"] = nn.Sequential(
                            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
                        )
            elif modality=='go':

                configuration = BertConfig(vocab_size=44261, max_position_embeddings=512)
                oneprot_model["go_model"] = GOBertModel(configuration)
                oneprot_model["go_head"] = nn.Sequential(
                            nn.LayerNorm(normalized_shape=go_hidden_size, eps=1e-6),
                            SelectElement(index=0),
                            nn.Dropout(p=0.5),
                            nn.Linear(go_hidden_size, out_embed_dim, bias=False),
                        )
                oneprot_model["go_post"] = nn.Sequential(
                            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
                        )
        self.oneprot_module = nn.ModuleDict(oneprot_model)


        loss_fn = losses.NTXentLoss()
        self.loss_fn = losses.SelfSupervisedLoss(loss_fn)
        #loss = loss_fn(embeddings, ref_emb)
        # loss function
        #self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, batch) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        
        
        modality_outputs = [] 
        sequence_outputs = []
        for modality in list(batch.keys()):
            
            sequence_features, modality_features = batch[modality]
            sequence_output = self.oneprot_module['sequence_model'](**sequence_features)
            output_head = self.oneprot_module["sequence_head"](sequence_output['last_hidden_state'])
            #sequence_outputs.extend(output_head)
            sequence_outputs.extend(self.oneprot_module["sequence_post"](output_head))
        
            #modality_outputs[f"{modality}_sequence_feat"]=sequence_output_pos    
                
            if modality == 'msa':    
                
                modality_output = self.oneprot_module[f"{modality}_model"](modality_features,repr_layers=[12])
                modality_output = modality_output['representations'][12][:,0,:,:]
                output_head = self.oneprot_module[f"{modality}_head"](modality_output)
                #modality_output_post = self.oneprot_module[f"{modality}_post"](output_head)
                #modality_outputs[f"{modality}_modality_feat"]=modality_output_post
                
            elif modality == 'structure': 
                    
                modality_output = self.oneprot_module[f"{modality}_model"](modality_features)
                output_head = self.oneprot_module[f"{modality}_head"](modality_output)
                #modality_output_post = self.oneprot_module[f"{modality}_post"](output_head)
                #modality_outputs[f"{modality}_modality_feat"]=modality_output_post
            
            elif modality == 'text':    
                modality_output = self.oneprot_module[f"{modality}_model"](**modality_features)
                output_head = self.oneprot_module[f"{modality}_head"](modality_output['last_hidden_state'])
                #modality_output_post = self.oneprot_module[f"{modality}_post"](output_head)
                #modality_outputs[f"{modality}_modality_feat"]=modality_output_post

            elif modality == 'go':    
          
                trunk_output = self.oneprot_module[f"{modality}_model"](**modality_features)
                output_head = self.oneprot_module[f"{modality}_head"](trunk_output['last_hidden_state'])
                #modality_output_post = self.oneprot_module[f"{modality}_post"](output_head)
                #modality_outputs[f"{modality}_modality_feat"]=modality_output_post
            
            modality_output_post = self.oneprot_module[f"{modality}_post"](output_head)
            modality_outputs.extend(modality_output_post)

        return torch.stack(sequence_outputs), torch.stack(modality_outputs)

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

        sequence_outputs, modality_outputs = self.forward(batch)


        all_sequence_outputs = self.all_gather(sequence_outputs, sync_grads=True).view(-1, sequence_outputs.shape[1])
        all_modality_outputs = self.all_gather(modality_outputs, sync_grads=True).view(-1, modality_outputs.shape[1])
    
        feats = torch.cat((all_sequence_outputs, all_modality_outputs), 0)
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / 0.07
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = nll.mean()

        # Logging loss

        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics

        #loss = self.loss_fn(all_sequence_outputs, all_modality_outputs)
        
        # update and log metrics
        self.train_loss(loss)
        #self.log('train_loss_ml', nll)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('train_acc_top1', (sim_argsort == 0).float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/acc_top10', (sim_argsort < 10).float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/acc_top50', (sim_argsort < 50).float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/acc_mean_pos', 1+sim_argsort.float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor] = None, batch_idx: int = 0, dataloader_idx: int = 0) -> None:

        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_data = {}
        batch_data[self.data_modalities[dataloader_idx]]= batch
        sequence_outputs, modality_outputs = self.forward(batch_data)
                
        all_sequence_outputs = self.all_gather(sequence_outputs).view(-1, sequence_outputs.shape[1])
        all_modality_outputs = self.all_gather(modality_outputs).view(-1, modality_outputs.shape[1])
    
        feats = torch.cat((all_sequence_outputs, all_modality_outputs), 0)
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / 0.07
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = nll.mean()

        # Logging loss

        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics

        #loss = self.loss_fn(all_sequence_outputs, all_modality_outputs)
        
        # update and log metrics
        self.val_loss(loss)
        #self.log('train_loss_ml', nll)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('val_acc_top1', (sim_argsort == 0).float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/acc_top10', (sim_argsort < 10).float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/acc_top50', (sim_argsort < 50).float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/acc_mean_pos', 1+sim_argsort.float().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
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

'''
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        return AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)

'''