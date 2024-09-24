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

class BaseEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        proj: str = None,
        use_logit_scale: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.proj = self._create_projection(proj)
        self.norm = self._create_normalization(use_logit_scale)

    def _create_projection(self, proj):
        if proj == 'linear':
            return nn.Linear(self.output_dim, self.output_dim, bias=False)
        elif proj == 'mlp':
            hidden_size = (self.output_dim + self.output_dim) // 2
            return nn.Sequential(
                nn.Linear(self.output_dim, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, self.output_dim, bias=False),
            )
        else:
            return nn.Identity()

    def _create_normalization(self, use_logit_scale):
        layers = [Normalize(dim=-1)]
        if use_logit_scale:
            layers.append(LearnableLogitScaling(learnable=True))
        else:
            layers.append(LearnableLogitScaling(learnable=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")