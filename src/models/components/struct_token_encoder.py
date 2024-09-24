import torch
import torch.nn as nn
from src.models.components.base_encoder import BaseEncoder
from transformers import AutoConfig, AutoModel
from src.models.components.pooling_layers import get_pooler

class StructTokenEncoder(BaseEncoder):
    def __init__(
        self,
        model_name_or_path: "esm2_t12_35M_UR50D",
        output_dim: int,
        pooler_type: str = "mean_pooler",
        proj: str = "linear",
        use_logit_scale: bool = False,
    ):
        super().__init__(output_dim, proj, use_logit_scale)
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.transformer = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.transformer.resize_token_embeddings(self.config.vocab_size + 21) # 21 is the number of newly added structure tokens
        self.pooler = get_pooler(pooler_type)(self.config.hidden_size, output_dim)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.pooler(outputs.last_hidden_state, attention_mask)
        projected = self.proj(pooled_output)
        return self.norm(projected)