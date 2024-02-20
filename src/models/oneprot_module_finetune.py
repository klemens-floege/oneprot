from typing import Any, Dict, Tuple
import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from torch import nn
from collections import Counter
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from transformers import StoppingCriteria, StoppingCriteriaList, EsmForProteinFolding
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
import copy
from src.models.components.sequence import SequenceModel
from src.models.components.structure import StructureModel
from dig.threedgraph.method import ProNet
from src.models.components.retrieval_metric import RetrievalMetric
from collections import OrderedDict
from torch.nn.utils import rnn
from src.models.components.utils_tokenizer import escape_custom_split_sequence
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    TaskType
)

from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs



class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
            if stop_count >= self.ENCOUNTERS:
                return True
        return False

class OpenOPTPEFTModel(LightningModule):

    def __init__(
        self,
        max_tgt_len: int,
        pretrained_lm: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,

    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super(OpenOPTPEFTModel, self).__init__()     

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        seq_model = SequenceModel(model_name_or_path='facebook/esm2_t33_650M_UR50D',
            output_dim=1024,
            pooler_type= 'mean_pooler',
            proj= 'linear',
            use_logit_scale= False)
         
        encoder = ProNet(level='aminoacid', out_channels=1024)
        structure_model = StructureModel(encoder, output_dim=1024,proj= 'linear',
                    use_logit_scale= False)
        oneprot = nn.ModuleDict({
                        'sequence': seq_model,
                        'struct': structure_model,
                })

        model_ckpt = torch.load('/p/project/hai_oneprot/merdivan1/oneprot/logs/train/runs/2024-01-30_07-46-47/checkpoints/epoch_016.ckpt', map_location=torch.device('cpu'))
        # Remove "oneprot." prefix from keys
        modified_state_dict = OrderedDict((key.replace('oneprot.', ''), value) for key, value in model_ckpt['state_dict'].items())
        oneprot.load_state_dict(modified_state_dict, strict=True)
        
        self.oneprot_structure = copy.deepcopy(oneprot['struct'])
        #self.oneprot_sequence = copy.deepcopy(oneprot['sequence'])
        # free vision encoder
        for name, param in self.oneprot_structure.named_parameters():
            param.requires_grad = False
        self.oneprot_structure.eval()
        
        #for name, param in self.oneprot_sequence.named_parameters():
        #    param.requires_grad = False
        #self.oneprot_sequence.eval()  
               
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=['embed_tokens', 'q_proj', 'k_proj', 'v_proj']
        )

        model = AutoModelForCausalLM.from_pretrained(pretrained_lm)
      
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_lm, use_fast=False)
        self.tokenizer.pad_token_id = 1
        self.tokenizer.pad_token = "<pad>"
        #self.tokenizer.padding_side = "left"

        # setup truncation
        #self.tokenizer.truncation_side = "left"

        # setup special tokens
        self.tokenizer.bos_token_id = 0
        self.tokenizer.bos_token = "<s>"

        self.tokenizer.eos_token_id = 2
        self.tokenizer.eos_token = "</s>"

        self.tokenizer.unk_token = "<unk>"
        self.tokenizer.unk_token_id = 3
        
        special_tokens_dict = {"additional_special_tokens": ['<Struct>','</Struct>','<Seq>','</Seq>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens = True)
        print ('Language decoder initialized.')

        
        model.resize_token_embeddings(len(self.tokenizer))
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        self.model.gradient_checkpointing_enable()
        self.proj_struct = nn.Linear(
            1024, self.model.config.hidden_size
        )
        
        #self.proj_sequence = nn.Linear(
        #    1024, self.model.config.hidden_size
        #)

        self.max_tgt_len = max_tgt_len
        
     
        #self.esm_fold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        #self.esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

        #for name, param in self.esm_model.named_parameters():
        #    param.requires_grad = False
        #self.esm_model.eval()



        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        # for averaging loss across batches
        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()
        self.test_acc = MeanMetric()

        # for tracking best so far validation loss
        self.val_acc_best = MaxMetric()
        
     
    
    def forward(self, inputs):
        

        modality_embs = {}
        if "structure" in inputs:
            input_struct = inputs['structure']
            struct_embeds, _ = self.encode_structure(input_struct)
            modality_embs["struct"] = struct_embeds
        
        if "sequence" in inputs:
            input_sequence = inputs['sequence']
            seq_embeds, _ = self.encode_sequence(input_sequence)
            modality_embs["sequence"] = seq_embeds

        instructions = inputs['instruction']
        responses = inputs['response']
        
        input_ids, target_ids, attention_mask = self.process_batch(instructions, responses)
        
        inputs_embeds, targets, attention_mask  = self.insert_modalities(modality_embs, input_ids, target_ids, attention_mask)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = acc & valid_mask    # [B*S]
        acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, acc


    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_loss_best.reset()
        
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        for task in list(batch.keys()):     
            
            inputs = batch[task]
            loss, acc = self.forward(inputs)
            self.train_loss(loss)
            self.train_acc(acc)
      
            self.log(f"train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
            
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss, acc = self.forward(batch)
        
        
        #generated_sequences = self.model.generate(batch)
        #print("Sample of generated sequence = ", generated_sequences[0])
        #sequence_embeds, _ = self.encode_sequence(generated_sequences)
        #self.metrics["val_struct_ret"].update(sequence_embeds, struct_embeds)

        self.val_loss(loss)
        self.val_acc(acc)
    
        self.log(f"val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor] = None, batch_idx: int = 0, dataloader_idx: int = 0) -> None:

        for task in list(batch.keys()):     
            
            inputs = batch[task]
            loss, acc = self.forward(inputs)
            
            self.test_loss(loss)
            self.test_acc(acc)
      
            self.log(f"test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        

    def on_test_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.test_loss.compute()  # get current val acc
        
        
        self.log("test/loss", loss, sync_dist=True, prog_bar=True)
        

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        

    def encode_structure(self, structure_inputs):

        with torch.no_grad():
            embeddings = self.oneprot_structure(structure_inputs) # bsz x 1024
     
        inputs_galactica = self.proj_struct(embeddings).unsqueeze(1) # bsz x 1 x llama_size
        atts_galatica = torch.ones(inputs_galactica.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_galactica, atts_galatica

    def encode_seq(self, seq_inputs):

        with torch.no_grad():
            embeddings = self.oneprot_sequence(seq_inputs) # bsz x 1024
     
        inputs_galactica = self.proj_sequence(embeddings).unsqueeze(1) # bsz x 1 x llama_size
        atts_galatica = torch.ones(inputs_galactica.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_galactica, atts_galatica
    
    def process_batch(self, raw_instructions, responses):
    
        batch_size = len(raw_instructions)
        targets = [escape_custom_split_sequence(x) for x in responses]
        instructions = [escape_custom_split_sequence(x) for x in raw_instructions]
        
        model_inputs = self.tokenizer(instructions, add_special_tokens=False)
        labels = self.tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
        batch_input_ids, batch_target_ids = [], []
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.eos_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            batch_input_ids.append(torch.LongTensor(model_inputs["input_ids"][i]))
            batch_target_ids.append(torch.LongTensor(labels["input_ids"][i]))


        input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
        input_ids = input_ids[:,:self.max_tgt_len]
        target_ids = target_ids[:,:self.max_tgt_len]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return input_ids, target_ids, attention_mask
    
    def insert_modalities(self, modality_embs, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = input_ids.size()[0]
        
        bos = torch.ones([batch_size, 1],
                         dtype=input_ids.dtype,
                         device=input_ids.device) * self.tokenizer.bos_token_id # bsz x 1
        
        self.model.to(self.device)
        # peft model need deeper call
        p_after_embeds = self.model.model.model.decoder.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim

        bos_embeds = self.model.model.model.decoder.embed_tokens(bos) # bsz x 1 x embed_dim
        if "struct" in modality_embs:
        
       
            start_token = self.tokenizer("<Struct>", return_tensors="pt", add_special_tokens=False).to(self.device)
            start_emb = self.model.model.model.decoder.embed_tokens(start_token.input_ids).expand(batch_size, -1, -1).to(self.device) # bsz x s1 x embed_di
            
            end_token = self.tokenizer("</Struct>", return_tensors="pt", add_special_tokens=False).to(self.device)
            end_emb = self.model.model.model.decoder.embed_tokens(end_token.input_ids).expand(batch_size, -1, -1).to(self.device) # bsz x s1 x embed_di
            modality_embs['struct'] = modality_embs['struct'].to(self.device)    
            inputs_embeds = torch.cat([bos_embeds, start_emb, modality_embs['struct'], end_emb, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        
        else:
            inputs_embeds = torch.cat([bos_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        # create targets
        empty_targets = (
            torch.ones([batch_size, 4], # 1 (bos) + <Struct> + struct emb + </Struct> # 1 (bos) + s1 + 1 (image vector)
                       dtype=torch.long).to(self.device).fill_(-100)  
        ) # bsz x (1 + s1 + 1)
        targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_prefix = torch.ones([batch_size, 4], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
        return inputs_embeds, targets, attention_mask 

    def insert_generation_modalities(self, model_inputs, modality_embeds):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        
        input_ids = torch.LongTensor(model_inputs["input_ids"]).to(self.device) # bsz x s2
      

        batch_size = input_ids.size()[0]
        
        bos = torch.ones([batch_size, 1],
                         dtype=input_ids.dtype,
                         device=input_ids.device) * self.tokenizer.bos_token_id # bsz x 1
        
        self.model.to(self.device)
        # peft model need deeper call
        p_after_embeds = self.model.model.model.decoder.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim

        bos_embeds = self.model.model.model.decoder.embed_tokens(bos) # bsz x 1 x embed_dim
        if "struct" in modality_embeds:
        
       
            start_token = self.tokenizer("<Struct>", return_tensors="pt", add_special_tokens=False).to(self.device)
            start_emb = self.model.model.model.decoder.embed_tokens(start_token.input_ids).expand(batch_size, -1, -1).to(self.device) # bsz x s1 x embed_di
            
            end_token = self.tokenizer("</Struct>", return_tensors="pt", add_special_tokens=False).to(self.device)
            end_emb = self.model.model.model.decoder.embed_tokens(end_token.input_ids).expand(batch_size, -1, -1).to(self.device) # bsz x s1 x embed_di
            modality_embeds['struct'] = modality_embeds['struct'].to(self.device)
            mod_emb = torch.cat([start_emb, modality_embeds['struct'], end_emb], dim=1)        
            inputs_embeds = torch.cat([bos_embeds, mod_emb, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        else:
            inputs_embeds = torch.cat([bos_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
       

        return inputs_embeds

    def prepare_generation_embedding(self, inputs):
        
        modality_embs = {}
        if "structure" in inputs:
            input_struct = inputs['structure']
            struct_embeds, _ = self.encode_structure(input_struct)
            modality_embs["struct"] = struct_embeds
        
        if "sequence" in inputs:
            input_sequence = inputs['sequence']
            seq_embeds, _ = self.encode_sequence(input_sequence)
            modality_embs["sequence"] = seq_embeds

            
        model_inputs = self.tokenizer(inputs['instruction'], add_special_tokens=False)         
        inputs_embeds  = self.insert_generation_modalities(model_inputs, modality_embs)    
            
        return inputs_embeds

    def generate(self, inputs):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''
        
        self.model.gradient_checkpointing_disable()
        input_embeds = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[18], encounters=1)])
        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=1024,
            top_p=0.7,
            temperature=0.01,
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.model.gradient_checkpointing_enable()
        return output_text