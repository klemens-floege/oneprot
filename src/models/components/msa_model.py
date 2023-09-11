""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType

from src.models.components.layers import LearnableLogitScaling, Normalize
import esm

class MsaModel(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            output_dim: int,
            proj: str = 'mlp',
            use_logit_scale: str = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.transformer, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()  
        self.transformer.eval() 
        for param in self.transformer.parameters():
            param.requires_grad = False
        d_model = 768
        if (d_model == output_dim) and (proj is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                #nn.Dropout(p=0.5),
                nn.Linear(hidden_size, output_dim, bias=False),
            )
        if use_logit_scale:
            self.norm = nn.Sequential(
                            Normalize(dim=-1), 
                            LearnableLogitScaling(learnable=True)
                    )
        else:
            self.norm = nn.Sequential(
                            Normalize(dim=-1), 
                    )

    def forward(self, x: TensorType):
        pooled_out = self.transformer(tokens=x, repr_layers=[12])
        pooled_out = pooled_out['representations'][12][:,0,0,:]
        projected = self.proj(pooled_out)
        normed = self.norm(projected) 
        return normed

    def init_parameters(self):
        pass
