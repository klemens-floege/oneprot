""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType
import numpy as np
from src.models.components.layers import LearnableLogitScaling, Normalize
from dig.threedgraph.method import ProNet


class StructModel(nn.Module):

    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            output_dim: int = 512,
            
            euler_noise: bool = True,
            data_augment_eachlayer: bool = True,
            dropout: float = 0.25,
            proj: str = None,
            use_logit_scale: str = None,
    ):
        super().__init__()


        self.output_dim = output_dim
        self.model = ProNet(level='allatom', out_channels = output_dim, euler_noise=euler_noise, data_augment_eachlayer=data_augment_eachlayer, dropout=dropout)
        d_model = output_dim
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

    def forward(self, batch: TensorType):
        
        pooled_out = self.model(batch)
        projected = self.proj(pooled_out)
        normed = self.norm(projected) 
        return normed

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.model.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

    def init_parameters(self):
        pass

