import torch
import torch.nn as nn
import numpy as np


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class LearnableLogitScaling(nn.Module):
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self):
        st = f"logit_scale_init={self.logit_scale_init},learnable={self.learnable}," \
             f" max_logit_scale={self.max_logit_scale}"
        return st


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if features.dim() == 2:
            return features
        if input_mask is not None:
            masked_features = features * input_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / input_mask.sum(dim=1, keepdim=True)
        else:
            mean_pooled_features = torch.mean(features, dim=1)
        return mean_pooled_features


class CLSTokenPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        return features[:, 0]


class BaseEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        proj_type: str = None,
        use_logit_scale: bool = False,
        pooling_type: str = 'mean',
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.pooling_type = pooling_type
        self.proj = self._create_projection(proj_type)
        self.norm = self._create_normalization(use_logit_scale)
        self.pooling = self._create_pooling(pooling_type)

    def _create_projection(self, proj_type):
        if (self.d_model == self.output_dim) and (proj_type is None):
            return nn.Sequential(
                nn.Identity(),
            )
        elif proj_type == 'linear':
            return nn.Sequential(
                nn.LayerNorm(self.output_dim),
                nn.Linear(self.d_model, self.output_dim, bias=False)              
            )
        elif proj_type == 'mlp':
            hidden_size = (self.d_model + self.output_dim) // 2
            return nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(self.d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.LayerNorm(self.output_dim),
                nn.Linear(hidden_size, self.output_dim, bias=False)     
            )
        else:
            return nn.Sequential(
                nn.Identity(),
            )

    def _create_normalization(self, use_logit_scale):
        layers = [Normalize(dim=-1)]
        if use_logit_scale:
            layers.append(LearnableLogitScaling(learnable=True))
        else:
            layers.append(LearnableLogitScaling(learnable=False))
        return nn.Sequential(*layers)

    def _create_pooling(self, pooling_type):
        if pooling_type == 'mean':
            return MeanPooling()
        elif pooling_type == 'cls':
            return CLSTokenPooling()
        else:
            return nn.Identity()

    def forward(self, x, input_mask=None):
        x = self.pooling(x, input_mask)
        x = self.proj(x)
        x = self.norm(x)
        return x