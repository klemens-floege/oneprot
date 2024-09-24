from src.models.components.base_encoder import BaseEncoder
import torch
class StructEncoder(BaseEncoder):
    def __init__(
        self,
        encoder: torch.nn.Module,
        output_dim: int,
        proj: str = None,
        use_logit_scale: bool = False,
        level: str = "backbone",
        euler_noise: bool = True,
        data_augment_eachlayer: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__(output_dim, proj, use_logit_scale)
        self.encoder = encoder

    def forward(self, batch):
        encoded = self.encoder(batch)
        projected = self.proj(encoded)
        return self.norm(projected)