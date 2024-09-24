from src.models.components.base_encoder import BaseEncoder
import esm
from src.models.components.pooling_layers import get_pooler

class MsaEncoder(BaseEncoder):
    def __init__(
        self,
        output_dim: int,
        pooler_type: str = "mean_pooler",
        proj: str = None,
        use_logit_scale: bool = False,
        use_all_msa: bool = False,
    ):
        super().__init__(output_dim, proj, use_logit_scale)
        self.transformer, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.pooler = get_pooler(pooler_type)(768, output_dim)
        self.use_all_msa = use_all_msa

    def forward(self, tokens):
        out = self.transformer(tokens, repr_layers=[12])
        attn_mask = (tokens != self.transformer.alphabet.padding_idx).long()
        
        if self.use_all_msa:
            # Use mean averaging for all tokens
            out = out['representations'][12]  # shape: (batch_size, num_sequences, sequence_length, hidden_dim)
            # Compute mean across all dimensions except the last (hidden_dim)
            pooled_out = (out * attn_mask.unsqueeze(-1)).sum(dim=(1, 2)) / attn_mask.sum(dim=(1, 2)).unsqueeze(-1)
            # pooled_out shape: (batch_size, hidden_dim)
        else:
            # Use only the 0th token and apply pooler (original behavior)
            out = out['representations'][12][:, 0, :, :]
            attn_mask = attn_mask[:, 0, :]
            pooled_out = self.pooler(out, attn_mask)
        
        projected = self.proj(pooled_out)
        return self.norm(projected)