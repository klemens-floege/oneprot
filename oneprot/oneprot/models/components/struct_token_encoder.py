import torch
import torch.nn as nn
from src.models.components.base_encoder import BaseEncoder
from transformers import AutoConfig, AutoModel

class StructTokenEncoder(BaseEncoder):
    def __init__(
        self,
        model_name_or_path: str = "esm2_t12_35M_UR50D",
        output_dim: int = 768,
        pooling_type: str = "mean",
        proj_type: str = "linear",
        use_logit_scale: bool = False,
        learnable_logit_scale: bool = False,
    ):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        super().__init__(
            d_model=self.config.hidden_size,
            output_dim=output_dim,
            proj_type=proj_type,
            use_logit_scale=use_logit_scale,
            learnable_logit_scale=learnable_logit_scale,
            pooling_type=pooling_type
        )
        
        self.transformer = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.transformer.resize_token_embeddings(self.config.vocab_size + 21)  # 21 is the number of newly added structure tokens

    def forward(self, input_ids):
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.pooling(outputs.last_hidden_state, attention_mask)
        projected = self.proj(pooled_output)
        return self.norm(projected)