# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor
import numpy as np
import torch
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

class RetrievalMetric(Metric):
    r"""Compute ` retrieval based on cosine similarity `_.

    .. math::
        cos_{sim}(x,y) = \frac{x \cdot y}{||x|| \cdot ||y||} =
        \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2}\sqrt{\sum_{i=1}^n y_i^2}}

    where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.
    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model in float tensor with shape ``(N,d)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values in float tensor with shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``score`` (:class:`~Dict`): A dictionary containing the keys ``precision``, ``recall`` and ``f1`` with
      corresponding values
    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.


    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        k: list = [1, 10],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `RetrievalMetric` will save all targets and predictions in the buffer."
            " For large datasets, this may lead to large memory footprint."
        )
        self.k = k
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Compute CLIP retrieval scores."""
        #print(self.preds,"sequence outputs!!!!")
        #print(self.preds.shape,"sequence outputs shape!!!!")

        #print(self.target,"modality outputs!!!!")
        #print(self.target.shape,"modality outputs shape!!!!")  
        sequence_outputs = dim_zero_cat(self.preds)
        modality_outputs = dim_zero_cat(self.target)
        
      

        metrics = {}
        logits_per_sequence = (sequence_outputs @ modality_outputs.t()).detach().cpu()
        logits_per_modality = logits_per_sequence.t().detach().cpu()
            
        logits = {"seq_to_mod": logits_per_sequence, "mod_to_seq": logits_per_modality}
        ground_truth = torch.arange(len(modality_outputs)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in self.k:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        #print(torch.distributed.get_rank()," I computed!!!")

        return metrics

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed


        """
        return self._plot(val, ax)