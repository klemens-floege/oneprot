import re

import torch
import torch.nn as nn
from torch import TensorType
import numpy as np
import collections
from unicore import checkpoint_utils
from unicore import tasks
from unimol.tasks.unimol_pocket import UniMolPocketTask
from unicore.data.dictionary import Dictionary
import os






from src.models.components.layers import LearnableLogitScaling, Normalize

class PocketModel(nn.Module):

    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            #encoder: torch.nn.Module,
            #task,
            #args,
            #dictionary,
            ckpt_path=None,
            output_dim: int = 512,
            proj: str = None,
            use_logit_scale: str = None, train_enc: bool =True,
            data_dir='/p/scratch/found/structures/swissprot/',
            task='unimol_pocket',
            train_subset='train', 
            valid_subset='val',
            curriculum=0,
            arch='unimol_base',
            tmp_save_dir='./', restore_file='checkpoint_last.pt', finetune_from_model=None,
            checkpoint_suffix='', mode='infer', 
            mask_prob=0.15, leave_unmasked_prob=0.05, random_token_prob=0.05, noise_type='uniform', 
            noise=1.0, remove_hydrogen=False, 
            remove_polar_hydrogen=False, max_atoms=256, dict_name='dict_coarse.txt', 
            no_seed_provided=False, encoder_layers=15, 
            encoder_embed_dim=512, encoder_ffn_embed_dim=2048, 
            encoder_attention_heads=64, dropout=0.1, emb_dropout=0.1, 
            attention_dropout=0.1, 
            activation_dropout=0.0, pooler_dropout=0.0, 
            max_seq_len=512, activation_fn='gelu', 
            pooler_activation_fn='tanh', post_ln=False,

    ):
        super().__init__()

        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        args=Namespace(
         seed=1,
         task=task,
         train_subset=train_subset, 
         valid_subset=valid_subset,
         curriculum=curriculum,
         arch=arch,
         tmp_save_dir=tmp_save_dir, restore_file=restore_file, finetune_from_model=finetune_from_model,
         checkpoint_suffix=checkpoint_suffix, mode=mode, data=data_dir, 
         mask_prob=mask_prob, leave_unmasked_prob=leave_unmasked_prob, random_token_prob=random_token_prob, noise_type=noise_type, 
         noise=noise, remove_hydrogen=remove_hydrogen, 
         remove_polar_hydrogen=remove_polar_hydrogen, max_atoms=max_atoms, dict_name=dict_name, 
         no_seed_provided=no_seed_provided, encoder_layers=encoder_layers, 
         encoder_embed_dim=encoder_embed_dim, encoder_ffn_embed_dim=encoder_ffn_embed_dim, 
         encoder_attention_heads=encoder_attention_heads, dropout=dropout, emb_dropout=emb_dropout, 
         attention_dropout=attention_dropout, 
         activation_dropout=activation_dropout, pooler_dropout=pooler_dropout, 
         max_seq_len=max_seq_len, activation_fn=activation_fn, 
         pooler_activation_fn=pooler_activation_fn, post_ln=post_ln)

        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        task=UniMolPocketTask(args, dictionary)
        task_pocket=task.setup_task(args)
        model=task_pocket.build_model(args)
        

        #print(ckpt_path,"ckpt path!!!!!!!!!!!!!!!!")
        if ckpt_path is not None:        
            #print(ckpt_path,"ckpt path!!!!!!!!!!!!!!!!")
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
            model.load_state_dict(state["model"], strict=False)
        
        self.encoder=model
        self.output_dim = output_dim
        d_model = args.encoder_embed_dim

        if (d_model == output_dim) and (proj is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        
        if use_logit_scale:
            self.norm = nn.Sequential(
                            Normalize(dim=-1), 
                            LearnableLogitScaling(learnable=True)
                    )
        else:
            self.norm = nn.Sequential(
                            Normalize(dim=-1), 
                            LearnableLogitScaling(learnable=False)
                    )
    
        if not train_enc:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, batch: collections.OrderedDict):
        
        src_tokens=batch['src_tokens']
        src_distance=batch['src_distance']
        src_coord=batch['src_coord']
        src_edge_type = batch['src_edge_type']


        pooled_out = self.encoder(src_tokens,src_distance,src_coord,src_edge_type)[0]

        projected = self.proj(pooled_out)
        normed = self.norm(projected) 

        normed=torch.mean(normed,1)
        normed=torch.squeeze(normed)

        return normed
        #return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.model.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

    def init_parameters(self):
        pass