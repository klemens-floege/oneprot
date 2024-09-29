from src.models.components.base_encoder import BaseEncoder
import torch
import torch.nn as nn

class StructEncoder(BaseEncoder):
    def __init__(
        self,
        encoder: torch.nn.Module,
        output_dim: int,
        proj_type: str = None,
        use_logit_scale: bool = False,
        learnable_logit_scale: bool = False,
        pooling_type: str = None,
        level: str = "backbone",
        euler_noise: bool = True,
        data_augment_eachlayer: bool = True,
        dropout: float = 0.25,
    ):
        # Assume encoder.output_dim is the dimension of the encoder's output
        super().__init__(
            d_model=output_dim,
            output_dim=output_dim,
            proj_type=proj_type,
            use_logit_scale=use_logit_scale,
            learnable_logit_scale=learnable_logit_scale,
            pooling_type=pooling_type
        )
        self.encoder = encoder
        
        # Store additional parameters
        self.level = level
        self.euler_noise = euler_noise
        self.data_augment_eachlayer = data_augment_eachlayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        encoded = self.encoder(batch)
        
        # Apply dropout
        encoded = self.dropout(encoded)    
        projected = self.proj(encoded)
        return self.norm(projected)