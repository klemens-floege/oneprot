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
        modality_features,
        sequence_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    
    if gather_with_grad:
        all_modality_features = torch.cat(torch.distributed.nn.all_gather(modality_features), dim=0)
        all_sequence_features = torch.cat(torch.distributed.nn.all_gather(sequence_features), dim=0)
    else:
        gathered_modality_features = [torch.zeros_like(modality_features) for _ in range(world_size)]
        gathered_sequence_features = [torch.zeros_like(sequence_features) for _ in range(world_size)]
        dist.all_gather(gathered_modality_features, modality_features)
        dist.all_gather(gathered_sequence_features, sequence_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_modality_features[rank] = modality_features
            gathered_sequence_features[rank] = sequence_features
        all_modality_features = torch.cat(gathered_modality_features, dim=0)
        all_sequence_features = torch.cat(gathered_sequence_features, dim=0)

    return all_modality_features, all_sequence_features

def ring_infonce(x_embs, y_embs, logit_scale):
    y_embs_send_buff = y_embs.clone()
    y_embs_recv_buff = y_embs.clone()

    local_logits = logit_scale * x_embs @ y_embs.T
    pos_pair_loss = -local_logits.diag()
    neg_pair_loss = torch.logsumexp(local_logits, dim=1, keepdim=True)

    size = dist.get_world_size()
    rank = dist.get_rank()
    left = ((rank - 1) + size) % size
    right = (rank + 1) % size
    for t in range(size - 1):
        if t % 2 == 0:
            # Send send_buff
            send_req = dist.isend(y_embs_send_buff, right)
            dist.recv(y_embs_recv_buff, left)
            cross_logits = logit_scale * x_embs @ y_embs_recv_buff.T
            neg_pair_loss[:] = torch.logsumexp(
                torch.cat((neg_pair_loss, cross_logits), dim=1), dim=1, keepdim=True
            )
        else:
            send_req = dist.isend(y_embs_recv_buff, right)
            dist.recv(y_embs_send_buff, left)
            cross_logits = logit_scale * x_embs @ y_embs_send_buff.T
            neg_pair_loss[:] = torch.logsumexp(
                torch.cat((neg_pair_loss, cross_logits), dim=1), dim=1, keepdim=True
            )
        send_req.wait()
    loss = pos_pair_loss + neg_pair_loss.squeeze()
    return loss.mean()

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_ring_reduce=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_ring_reduce = use_ring_reduce  
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

    def get_logits(self, modality_features, sequence_features, logit_scale):
        if self.world_size > 1:
            all_modality_features, all_sequence_features = gather_features(
                modality_features, sequence_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size)
          
            if self.local_loss:
                logits_per_modality = logit_scale * modality_features @ all_sequence_features.T
                logits_per_sequence = logit_scale * sequence_features @ all_modality_features.T
            else:
                logits_per_modality = logit_scale * all_modality_features @ all_sequence_features.T
                logits_per_sequence = logits_per_modality.T
        else:
            logits_per_modality = logit_scale * modality_features @ sequence_features.T
            logits_per_sequence = logit_scale * sequence_features @ modality_features.T
        
        return logits_per_modality, logits_per_sequence

    def forward(self, modality_features, sequence_features, logit_scale=1.0, output_dict=False):
        device = modality_features.device
        
        if self.use_ring_reduce:
            modality_loss = ring_infonce(modality_features, sequence_features, logit_scale)
            sequence_loss = ring_infonce(sequence_features, modality_features, logit_scale)
            total_loss = (modality_loss + sequence_loss) / 2
        else:
            logits_per_modality, logits_per_sequence = self.get_logits(modality_features, sequence_features, logit_scale)

            labels = self.get_ground_truth(device, logits_per_modality.shape[0])

            total_loss = (
                F.cross_entropy(logits_per_modality, labels) +
                F.cross_entropy(logits_per_sequence, labels)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

        
        
        

