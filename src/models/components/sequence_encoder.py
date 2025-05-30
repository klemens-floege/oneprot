import torch
import torch.nn as nn
from torch import TensorType
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model
from src.models.components.base_encoder import BaseEncoder

def create_model_function(pretrained, local_only=True):
    def func(model_name_or_path, config=None, **kwargs):
        if pretrained:
            return AutoModel.from_pretrained(
                model_name_or_path,
                cache_dir="/path/to/your/cache/folder",
                local_files_only=local_only,
                **kwargs
            )
        else:
            return AutoModel.from_config(config, **kwargs)
    return func


class SequenceEncoder(BaseEncoder):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        pooling_type: str = "mean",
        proj_type: str = None,
        use_logit_scale: bool = False,
        learnable_logit_scale: bool = False,
        pretrained: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list = ["query", "key", "value"],
        frozen: bool = True
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
        #config = AutoConfig.from_pretrained("path/to/downloaded/config.json")
        #self.transformer = AutoModel.from_pretrained("path/to/downloaded/pytorch_model.bin", config=config)
        #create_func = AutoModel.from_pretrained if pretrained else AutoModel.from_config
        create_func = create_model_function(pretrained=pretrained, local_only=False)
        self.transformer = create_func(
            model_name_or_path if pretrained else self.config,
            add_pooling_layer=False
        )
        #self.transformer.gradient_checkpointing_enable()
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
            bias="all",
            )
            self.transformer = get_peft_model(self.transformer, peft_config)

    def forward(self, x: TensorType):
        attention_mask = (x != self.config.pad_token_id).long()
        outputs = self.transformer(input_ids=x, attention_mask=attention_mask)
        pooled_output = self.pooling(outputs.last_hidden_state, attention_mask)
        projected = self.proj(pooled_output)
        return self.norm(projected)
