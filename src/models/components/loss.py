
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


def gather_features(
        sequence_features,
        modality_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    
    # We gather tensors from all gpus
    if gather_with_grad:
        all_sequence_features = torch.cat(torch.distributed.nn.all_gather(sequence_features), dim=0)
        all_modality_features = torch.cat(torch.distributed.nn.all_gather(modality_features), dim=0)
    else:
        gathered_sequence_features = [torch.zeros_like(sequence_features) for _ in range(world_size)]
        gathered_modality_features = [torch.zeros_like(modality_features) for _ in range(world_size)]
        dist.all_gather(gathered_sequence_features, sequence_features)
        dist.all_gather(gathered_modality_features, modality_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_sequence_features[rank] = sequence_features
            gathered_modality_features[rank] = modality_features
        all_sequence_features = torch.cat(gathered_sequence_features, dim=0)
        all_modality_features = torch.cat(gathered_modality_features, dim=0)

    return all_sequence_features, all_modality_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, sequence_features, modality_features):
        #print(sequence_features.shape, modality_features.shape," shapes of sequence_features, modality_features!!!!!!!!!!")
        if self.world_size > 1:
            all_sequence_features, all_modality_features = gather_features(
                sequence_features, modality_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_sequence =  sequence_features @ all_modality_features.T
                logits_per_modality =  modality_features @ all_sequence_features.T
            else:
                logits_per_sequence =  all_sequence_features @ all_modality_features.T
                logits_per_modality = logits_per_sequence.T
        else:
            logits_per_sequence =  sequence_features @ modality_features.T
            logits_per_modality =  modality_features @ sequence_features.T
        
        return logits_per_sequence, logits_per_modality

    def forward(self, sequence_features, modality_features,  output_dict=False):
        device = sequence_features.device
        logits_per_sequence, logits_per_modality = self.get_logits(sequence_features, modality_features)

        labels = self.get_ground_truth(device, logits_per_sequence.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_sequence, labels) +
            F.cross_entropy(logits_per_modality, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, sequence_features, modality_features, logit_bias=None):
        logits =  sequence_features @ modality_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, sequence_features, modality_features, logit_bias=None, negative_only=False):
        logits = self.get_logits(sequence_features, modality_features, logit_bias)
        labels = self.get_ground_truth(
            sequence_features.device,
            sequence_features.dtype,
            sequence_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / sequence_features.shape[0]
        return loss

    def forward(self, sequence_features, modality_features, logit_bias=None, output_dict=False):
        loss = self._loss(sequence_features, modality_features, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                modality_features_to_right = modality_features_to_left = modality_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    modality_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        modality_features_to_left,
                        modality_features_to_right,
                    )

                    for f in modality_features_recv:
                        loss += self._loss(
                            sequence_features,
                            f,
                            logit_bias,
                            negative_only=True,
                        )
                    modality_features_to_left, modality_features_to_right = modality_features_recv

                if remainder:
                    modality_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, modality_features_to_right)

                    loss += self._loss(
                        sequence_features,
                        modality_features_recv,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                modality_features_to_right = modality_features
                for i in range(self.world_size - 1):
                    modality_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, modality_features_to_right)

                    loss += self._loss(
                        sequence_features,
                        modality_features_from_left,
                        logit_bias,
                        negative_only=True,
                    )
                    modality_features_to_right = modality_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
        

