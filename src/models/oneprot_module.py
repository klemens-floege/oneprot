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
        

        self.metrics = {
            f"{split}_{modality}": RetrievalMetric()
            for split in ["val", "test"]
            for modality in list(self.network.keys()) + ["seqsim", "seqsim_msa"]
            if modality != 'sequence'
        }

        # Storage for embeddings and sequences during testing
        self.test_embeddings = defaultdict(list)
        self.test_sequences = []
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

        if modality in ["sequence", "seqsim", "seqsim_msa"]:
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
            self.log(f"train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

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
        """
        Perform a test step.

        This method processes a batch of data for all modalities, computes embeddings,
        and stores them for later use in metric computation.

        Args:
            batch (Dict[str, Tuple]): Batch of data for each modality.
            batch_idx (int): Index of the current batch.

        Example:
            >>> batch = {
            ...     "sequence": (seq_inputs, seq_targets, _, sequences),
            ...     "structure": (struct_inputs, struct_targets, _, sequences)
            ... }
            >>> model.test_step(batch, 0)
        """
        for modality, (seq_inputs, mod_inputs, _, sequences) in batch.items():
            seq_features = self(seq_inputs, "sequence")
            mod_features = self(mod_inputs, modality)
            
            self.test_embeddings[f"sequence_{modality}"].append(seq_features.detach())
            self.test_embeddings[modality].append(mod_features.detach())
            
            if modality == list(batch.keys())[0]:
                self.test_sequences.extend(sequences)

            loss = self.loss_fn(seq_features, mod_features)
            self.test_loss(loss)
            self.log(f"test/loss_{modality}", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        """
        Compute and log retrieval metrics at the end of the test epoch.

        This method gathers embeddings from all processes, aligns them,
        and computes retrieval metrics for all modality pairs.

        Example:
            This method is called automatically by PyTorch Lightning at the end of the test epoch.
            It will compute and log metrics like:
            test/sequence_to_structure/R@1: 0.85
            test/structure_to_sequence/R@10: 0.95
            test/sequence_to_text/R@100: 0.99
        """
        gathered_embeddings = self.all_gather(self.test_embeddings)
        gathered_sequences = self.all_gather(self.test_sequences)
        
        combined_embeddings = defaultdict(list)
        all_sequences = []
        for process_embeddings, process_sequences in zip(gathered_embeddings, gathered_sequences):
            for modality, embeddings in process_embeddings.items():
                combined_embeddings[modality].extend(embeddings)
            all_sequences.extend(process_sequences)
        
        seq_to_idx = {seq: idx for idx, seq in enumerate(all_sequences)}
        combined_embeddings = {
            modality: torch.cat(embeddings, dim=0)
            for modality, embeddings in combined_embeddings.items()
        }
        
        aligned_embeddings = self.align_embeddings(combined_embeddings, seq_to_idx)

        self.compute_and_log_similarities(aligned_embeddings)

        self.test_embeddings.clear()
        self.test_sequences.clear()

    def align_embeddings(self, embeddings: Dict[str, torch.Tensor], seq_to_idx: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Align embeddings based on the sequence order.

        Args:
            embeddings (Dict[str, torch.Tensor]): Embeddings for each modality.
            seq_to_idx (Dict[str, int]): Mapping from sequence to index.

        Returns:
            Dict[str, torch.Tensor]: Aligned embeddings.

        Example:
            >>> embeddings = {
            ...     "sequence": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            ...     "structure": torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
            ... }
            >>> seq_to_idx = {"ACGT": 2, "TGCA": 0, "GACT": 1}
            >>> aligned = model.align_embeddings(embeddings, seq_to_idx)
            >>> print(aligned["sequence"])
            tensor([[0.3, 0.4],
                    [0.5, 0.6],
                    [0.1, 0.2]])
        """
        return {
            modality: emb[torch.tensor([seq_to_idx[seq] for seq in self.test_sequences], device=emb.device)]
            for modality, emb in embeddings.items()
        }

    def compute_and_log_similarities(self, embeddings: Dict[str, torch.Tensor]):
        """
        Compute and log retrieval metrics for all modality pairs.

        Args:
            embeddings (Dict[str, torch.Tensor]): Aligned embeddings for each modality.

        Example:
            >>> embeddings = {
            ...     "sequence_structure": torch.randn(100, 512),
            ...     "structure": torch.randn(100, 512),
            ...     "sequence_text": torch.randn(100, 512),
            ...     "text": torch.randn(100, 512)
            ... }
            >>> model.compute_and_log_similarities(embeddings)
            # This will log metrics like:
            # test/sequence_to_structure/R@1: 0.85
            # test/structure_to_sequence/R@10: 0.95
            # test/sequence_to_text/R@100: 0.99
            # test/text_to_sequence/R@1: 0.80
            # test/structure_to_text/R@10: 0.90
            # test/text_to_structure/R@100: 0.98
        """
        modalities = [mod for mod in embeddings.keys() if not mod.startswith("sequence_")]
        
        for mod in modalities:
            seq_mod = f"sequence_{mod}"
            sim_matrix = torch.matmul(embeddings[seq_mod], embeddings[mod].t())
            self.compute_and_log_retrieval_metrics(f"sequence_to_{mod}", sim_matrix)
            self.compute_and_log_retrieval_metrics(f"{mod}_to_sequence", sim_matrix.t())

        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                sim_matrix = torch.matmul(embeddings[mod1], embeddings[mod2].t())
                self.compute_and_log_retrieval_metrics(f"{mod1}_to_{mod2}", sim_matrix)
                self.compute_and_log_retrieval_metrics(f"{mod2}_to_{mod1}", sim_matrix.t())

    def compute_and_log_retrieval_metrics(self, pair: str, sim_matrix: torch.Tensor):
        """
        Compute and log retrieval metrics for a given similarity matrix.

        Args:
            pair (str): The modality pair (e.g., "sequence_to_structure").
            sim_matrix (torch.Tensor): Similarity matrix between two sets of embeddings.

        Example:
            >>> sim_matrix = torch.tensor([[1.0, 0.8, 0.6],
            ...                            [0.7, 1.0, 0.9],
            ...                            [0.5, 0.7, 1.0]])
            >>> model.compute_and_log_retrieval_metrics("sequence_to_structure", sim_matrix)
            # This will log:
            # test/sequence_to_structure/R@1: 1.0
            # test/sequence_to_structure/R@10: 1.0
            # test/sequence_to_structure/R@100: 1.0
        """
        k_values = [1, 10, 100]
        ground_truth = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        
        for k in k_values:
            top_k = torch.topk(sim_matrix, min(k, sim_matrix.shape[1]), dim=1)[1]
            correct = (top_k == ground_truth.unsqueeze(1)).any(dim=1)
            value = correct.float().mean().item()
            self.log(f"test/{pair}/R@{k}", value, sync_dist=True)


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