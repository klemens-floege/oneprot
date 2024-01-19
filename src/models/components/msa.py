""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType

from src.models.components.layers import LearnableLogitScaling, Normalize
import esm

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions

except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: TensorType, attention_mask: TensorType):
        masked_output = x * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: TensorType, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x[:, self.cls_token_position, :]



class MsaModel(nn.Module):

    def __init__(
            self,
            output_dim: int,
            proj: str = None,
            pooler_type: str= 'cls_pooler',
            use_logit_scale: str = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        
        self.pooler = _POOLERS[pooler_type]()
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
                            LearnableLogitScaling(learnable=False)
                    )

    def forward(self, x: TensorType):

        out = self.transformer(tokens=x, repr_layers=[12])
        attn_mask = (x != self.transformer.padding_idx).long()
        out = out['representations'][12][:,0,:,:]
        attn_mask = attn_mask[:,0,:]
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)
        normed = self.norm(projected) 
        return normed

    def init_parameters(self):
        pass
