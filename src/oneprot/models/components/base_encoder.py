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

class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1dPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_size = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_size, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_size, -1).bool(), float("-inf")
            )
        attn = torch.nn.functional.softmax(attn, dim=-1).view(batch_size, -1, 1)
        out = (attn * x).sum(dim=1)
        return out

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
        learnable_logit_scale: bool = False,
        pooling_type: str = 'mean',
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.pooling_type = pooling_type
        self.proj = self._create_projection(proj_type)
        self.norm = self._create_normalization(use_logit_scale, learnable_logit_scale)
        self.pooling = self._create_pooling(pooling_type)

    def _create_projection(self, proj_type):
        if (self.d_model == self.output_dim) and (proj_type is None):
            return nn.Sequential(
                nn.Identity(),
            )
        elif proj_type == 'linear':
            return nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.output_dim, bias=False)              
            )
        elif proj_type == 'mlp':
            hidden_size = (self.d_model + self.output_dim) // 2
            return nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, self.output_dim, bias=False)     
            )
        else:
            return nn.Sequential(
                nn.Identity(),
            )

    def _create_normalization(self, use_logit_scale, learnable_logit_scale=False):
        layers = [Normalize(dim=-1)]
        if use_logit_scale:
            if learnable_logit_scale:
                layers.append(LearnableLogitScaling(learnable=True))
            else:
                layers.append(LearnableLogitScaling(learnable=False))
        return nn.Sequential(*layers)

    def _create_pooling(self, pooling_type, hidden_size=1280):
        if pooling_type == 'mean':
            return MeanPooling()
        elif pooling_type == 'cls':
            return CLSTokenPooling()
        elif pooling_type == 'attention1d':
            return Attention1dPooling(hidden_size)
        else:
            return nn.Identity()

    def forward(self, x, input_mask=None):
        x = self.pooling(x, input_mask)
        x = self.proj(x)
        x = self.norm(x)
        return x