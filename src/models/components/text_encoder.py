import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.models.components.base_encoder import BaseEncoder
from src.models.components.pooling_layers import get_pooler
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Optional

class TextEncoder(BaseEncoder):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        pooler_type: str = "mean_pooler",
        proj: str = "linear",
        use_logit_scale: bool = False,
        frozen: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__(output_dim, proj, use_logit_scale)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        
        if frozen:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        if use_lora:
            if lora_target_modules is None:
                # Default target modules if none specified
                lora_target_modules = ["query", "key", "value"]
            
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.transformer = get_peft_model(self.transformer, peft_config)
        
        self.pooler = get_pooler(pooler_type)(self.config.hidden_size, output_dim)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.pooler(outputs.last_hidden_state, attention_mask)
        projected = self.proj(pooled_output)
        return self.norm(projected)