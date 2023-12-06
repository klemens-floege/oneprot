""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType
import numpy as np
from src.models.components.layers import LearnableLogitScaling, Normalize

try:
    import transformers
    from transformers import BertModel, AutoTokenizer, BertConfig, PretrainedConfig
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

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]

class GoModel(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            output_dim: int,
            num_attention_heads: int=10,
            num_hidden_layers: int=10,
            max_position_embeddings: int=300,
            hidden_size: int=200,
            proj: str = 'mlp',
            pooler_type: str= 'cls_pooler',
            use_logit_scale: str = None,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Initializing a BERT bert-base-uncased style configuration
        self.config = BertConfig(vocab_size=44263, pad_token_id=1, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers, max_position_embeddings=max_position_embeddings, hidden_size=hidden_size)

        # Initializing a model (with random weights) from the bert-base-uncased style configuration
        self.transformer = BertModel(self.config)
        self.transformer.embeddings.position_embeddings.weight.data = torch.zeros((self.config.max_position_embeddings, self.config.hidden_size))
        self.transformer.embeddings.position_embeddings.requires_grad_ = False
        
        with open('/p/scratch/hai_oneprot/go_emb_vals.npy', 'rb') as f:
            go_emb_vals = np.load(f)
        
        #self.transformer.embeddings.word_embeddings.weight.data  = torch.FloatTensor(go_emb_vals)
        
        self.pooler = _POOLERS[pooler_type]()
        
        d_model = hidden_size
        if (d_model == output_dim) and (proj is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
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
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)
        normed = self.norm(projected) 
        return normed

    def init_parameters(self):
        pass
