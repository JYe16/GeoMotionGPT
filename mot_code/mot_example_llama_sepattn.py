"""
LLaMA-based Motion-of-Thought (MoT) LM Head model.

This module provides the LLaMA version of the MoT model with language modeling head,
similar to mot_example_gpt2_sepattn.py but adapted for LLaMA architecture.
"""

import math
import os
import torch
from torch import nn
from typing import Optional, Tuple, List, Union

from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from .my_modeling_mot_llama_sepattn import MoTLlamaModel
from .modality_utils_sepattn import get_modalities_infos
from .mot_module import get_embeds_from_ids

import random
import numpy as np

logger = logging.get_logger(__name__)


def seed_setting(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MoTLlamaForCausalLM(LlamaForCausalLM):
    """
    LLaMA-based Motion-of-Thought model for causal language modeling.
    
    This model extends LlamaForCausalLM to support multi-modality (text + motion)
    with separate attention mechanisms for different modalities.
    """
    
    def __init__(
        self, 
        config: LlamaConfig, 
        modality_num: int = 2, 
        motion_codebook_size: int = 512 + 4,
        mot_factor: float = 1.0,
        attention_mode: str = 'all',
    ):
        # Initialize parent class first
        super().__init__(config)
        
        self.modality_num = modality_num
        self.forward_mod = 0  # 'text'
        self.config.d_model = config.hidden_size
        self.config.motion_vocab_size = motion_codebook_size
        self.config.mot_loss = 'skip'
        self.config.is_encoder_decoder = False  # LLaMA is decoder-only
        self.modality_infos = None
        self.last_pos_ids = None
        
        self.mot_factor = mot_factor
        # Align motion embedding dim with attention head structure
        self.mot_embed_dim = int(config.hidden_size // config.num_attention_heads * mot_factor) * config.num_attention_heads
        config.mot_embed_dim = self.mot_embed_dim
        config.embed_dims = [config.hidden_size, self.mot_embed_dim]
        config.mot_factor = mot_factor
        
        # Setup cross-modality attention layers
        all_layers_idx = list(range(config.num_hidden_layers))
        config.text_cross_model_attention = all_layers_idx[-1:]
        
        if attention_mode == 'all':
            config.cross_model_attention = all_layers_idx
        elif attention_mode == 'first':
            config.cross_model_attention = all_layers_idx[:1]
        elif attention_mode == 'last':
            config.cross_model_attention = all_layers_idx[-1:]
        elif attention_mode == 'Ahalf':
            config.cross_model_attention = all_layers_idx[:config.num_hidden_layers//2]
        elif attention_mode == 'halfB':
            config.cross_model_attention = all_layers_idx[-config.num_hidden_layers//2:]
        elif attention_mode == 'firstthird':
            config.cross_model_attention = all_layers_idx[:config.num_hidden_layers//3]
        elif attention_mode == 'midthird':
            config.cross_model_attention = all_layers_idx[config.num_hidden_layers//3:int(config.num_hidden_layers/3*2)]
        elif attention_mode == 'lastthird':
            config.cross_model_attention = all_layers_idx[int(config.num_hidden_layers/3*2):]
        elif attention_mode == 'odd':
            config.cross_model_attention = all_layers_idx[::2]
        elif attention_mode == 'even':
            config.cross_model_attention = all_layers_idx[1::2]
        elif attention_mode.startswith('last'):
            lnum = int(attention_mode.split('last')[-1])
            config.cross_model_attention = all_layers_idx[-lnum:]
        elif attention_mode.startswith('first'):
            lnum = int(attention_mode.split('first')[-1])
            config.cross_model_attention = all_layers_idx[:lnum]
        elif attention_mode == 'None':
            config.cross_model_attention = []
        else:
            raise ValueError(f'Unrecognized attention_mode: {attention_mode}')
        
        # Replace the model with MoT version
        self.model = MoTLlamaModel(config, modality_num=modality_num, mot_factor=mot_factor)
        
        # LM head for text tokens
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()
        
    def set_modality_info(self, tokenizer):
        """Setup modality information from tokenizer."""
        modality_infos = get_modalities_infos(config=self.config, tokenizer=tokenizer)
        self.mod2id = {m.modality_name: i for i, m in enumerate(modality_infos)}
        self.text_id = self.mod2id['text']
        
        modality_infos[self.text_id].pre_processor = self.model.embed_tokens
        modality_infos[self.text_id].post_processor = self.lm_head
        self.modality_infos = modality_infos
        self.pad_ids = [m.pad_id for m in modality_infos]
        
        pre_processors = [m.pre_processor for m in self.modality_infos]
        post_processors = [m.post_processor for m in self.modality_infos]
        self.pre_processors = nn.ModuleList(pre_processors)
        self.post_processors = nn.ModuleList(post_processors)
        
    def update_typeids(self, type_ids):
        """Update modality type IDs for all layers."""
        self.valid_pos = type_ids
        self.model.update_typeids(type_ids)
        
    def init_mod_token_embeddings(self, mod_id, added_num_tokens=None):
        """Initialize motion token embeddings from text embeddings."""
        if added_num_tokens is None:
            added_num_tokens = self.modality_infos[mod_id].mod_voc_size
            
        old_embeddings = self.pre_processors[0]
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_embeddings = self.pre_processors[mod_id]
        
        self._init_added_embeddings_weights_with_mean(
            old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
        )
        
        old_lm_head = self.post_processors[1]
        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size()
        new_lm_head = self.post_processors[mod_id]
        
        self._init_added_lm_head_weights_with_mean(
            old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed=False
        )
    
    def _init_added_embeddings_weights_with_mean(
        self, old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
    ):
        """Initialize new embedding weights with mean of existing embeddings."""
        with torch.no_grad():
            mean_embedding = old_embeddings.weight[:old_num_tokens].mean(dim=0)
            for i in range(new_embeddings.weight.shape[0]):
                new_embeddings.weight[i] = mean_embedding[:new_embeddings.weight.shape[1]]
                
    def _init_added_lm_head_weights_with_mean(
        self, old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed=False
    ):
        """Initialize new LM head weights with mean of existing weights."""
        with torch.no_grad():
            mean_weight = old_lm_head.weight[:old_num_tokens].mean(dim=0)
            for i in range(new_lm_head.weight.shape[0]):
                new_lm_head.weight[i] = mean_weight[:new_lm_head.weight.shape[1]]
        
    def get_embeddings_from_ids(self, input_ids, type_ids):
        """Get embeddings for each modality from input IDs."""
        inputs_embeds = []
        for i in range(self.modality_num):
            mod_valid_pos = (type_ids == i)
            mot_input_id = input_ids.masked_fill(~mod_valid_pos, self.pad_ids[i])
            mod_inputs_embeds = self.pre_processors[i](mot_input_id)
            inputs_embeds.append(mod_inputs_embeds)
        return inputs_embeds
    
    def forward(
        self,
        type_ids: torch.Tensor = None,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_mod: np.ndarray = None,
        mot_labels: torch.Tensor = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass with multi-modality support.
        
        Args:
            type_ids: Tensor indicating which modality each token belongs to
            input_ids: Input token IDs
            inputs_embeds: Pre-computed input embeddings (list of tensors for each modality)
            position_ids: Position IDs for RoPE
            attention_mask: Attention mask
            past_key_values: Cached key-value pairs for generation
            labels: Labels for language modeling loss
            mot_labels: Labels for motion tokens
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            forward_mod: Which modality to forward (deprecated)
            cache_position: Cache position for generation
        """
        self.type_ids = type_ids
        
        if type_ids is not None:
            valid_pos = torch.stack([type_ids == i for i in range(self.modality_num)], dim=0).to(type_ids.device)
            self.update_typeids(valid_pos)
        
        self.model.position_ids = position_ids
        
        if forward_mod is None:
            forward_mod = self.forward_mod
            
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = get_embeds_from_ids(input_ids, self.valid_pos, self.pad_ids, self.pre_processors)
            input_ids = None
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        
        # Get logits for each modality
        if isinstance(hidden_states, list):
            # Multi-modality output - compute logits per modality like GPT2
            lm_logits = [self.post_processors[i](hid) for i, hid in enumerate(hidden_states)]
        else:
            # Single hidden state - just use lm_head
            lm_logits = [self.lm_head(hidden_states)]
        
        # Compute loss using modality-specific loss functions (like GPT2 version)
        total_loss = None
        if labels is not None:
            total_loss = 0.
            if not isinstance(labels, list):
                mlabels = [labels] * self.modality_num
            else:
                mlabels = labels
                
            for i, out_logit in enumerate(lm_logits):
                if i >= len(self.valid_pos) or self.valid_pos[i].sum() == 0:
                    continue
                mlabel = mlabels[i].to(out_logit.device)
                loss_fct = self.modality_infos[i].loss_fct
                loss_mod = loss_fct(out_logit, mlabel, self.valid_pos[i])
                total_loss = total_loss + loss_mod.nan_to_num_(0)
            
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
            
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        type_ids=None,
        **kwargs,
    ):
        """Prepare inputs for generation step."""
        # If past_key_values is provided, only use the last token
        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if type_ids is not None:
                type_ids = type_ids[:, -1:]
                
        # Handle position_ids
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]
                
        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "type_ids": type_ids,
        }
        
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
            
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


if __name__ == "__main__":
    # Test code
    model_config = "deps/llama3.2-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = LlamaConfig.from_pretrained(model_config)
    print(f"LLaMA config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # Create MoT LLaMA model
    model = MoTLlamaForCausalLM(config, mot_factor=1.0).eval()
    
    # Add motion tokens to tokenizer
    tokenizer.add_tokens([f'<motion_id_{i}>' for i in range(512)])
    tokenizer.add_special_tokens({
        'additional_special_tokens': [
            '<start_of_motion>', '<end_of_motion>', 
            '<masked_motion>', '<pad_motion>'
        ]
    })
    
    model.config.mot_lm_dim = model.config.motion_vocab_size
    model.set_modality_info(tokenizer)
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    print("MoT LLaMA model created successfully!")
