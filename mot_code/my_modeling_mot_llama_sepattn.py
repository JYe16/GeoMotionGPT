"""
LLaMA-based Motion-of-Thought (MoT) model with separate attention for motion and text modalities.

This module adapts the MoT architecture from GPT2 to LLaMA 3.2.
Key differences from GPT2:
- Uses RMSNorm instead of LayerNorm
- Uses SiLU activation instead of GELU
- Uses RoPE (Rotary Position Embedding)
- Uses GQA (Grouped Query Attention)
- Different hidden size (2048 for LLaMA 1B vs 768 for GPT2-small)
"""

import math
import torch
from torch import nn
from typing import Optional, Tuple, List, Union

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import LlamaModel, LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from .mot_module import MoTDiffFuncMod, get_embeds_from_ids

logger = logging.get_logger(__name__)


class MoTLlamaRMSNorm(nn.Module):
    """MoT version of RMSNorm that handles multiple modalities."""
    
    def __init__(self, norms: List[nn.Module], modality_num: int = 2):
        super().__init__()
        self.fn = nn.ModuleList(norms)
        self.modality_num = modality_num
        self.valid_pos = None
        
    def update_typeids(self, valid_pos):
        self.valid_pos = valid_pos
        
    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i, hid in enumerate(hidden_states):
            outputs.append(self.fn[i](hid))
        return outputs


class MoTLlamaMLP(nn.Module):
    """MoT version of LlamaMLP for different modality dimensions."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoTLlamaAttention(nn.Module):
    """Multi-headed attention for MoT LLaMA with support for different modality dimensions."""
    
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        embed_dims: List[int],
        modality_num: int = 2,
        shared_attn: bool = True,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.modality_num = modality_num
        self.shared_attn = shared_attn
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.embed_dims = embed_dims
        self.valid_pos = None
        
        # Create projections for each modality
        if shared_attn:
            # Shared attention: all modalities project to same QKV space
            self.q_proj = nn.ModuleList([
                nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
                for dim in embed_dims
            ])
            self.k_proj = nn.ModuleList([
                nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=False)
                for dim in embed_dims
            ])
            self.v_proj = nn.ModuleList([
                nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=False)
                for dim in embed_dims
            ])
            self.o_proj = nn.ModuleList([
                nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
                for dim in embed_dims
            ])
        else:
            # Separate attention: each modality has its own attention head count
            self.q_proj = nn.ModuleList([
                nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
                for dim in embed_dims
            ])
            self.k_proj = nn.ModuleList([
                nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=False)
                for dim in embed_dims
            ])
            self.v_proj = nn.ModuleList([
                nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=False)
                for dim in embed_dims
            ])
            self.o_proj = nn.ModuleList([
                nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
                for dim in embed_dims
            ])
        
        # RoPE
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
    def update_typeids(self, valid_pos):
        self.valid_pos = valid_pos
        
    def forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.shared_attn:
            return self._forward_shared(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, cache_position
            )
        else:
            return self._forward_separate(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, cache_position
            )
    
    def _forward_shared(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """Forward with shared attention across modalities."""
        bsz, q_len, _ = hidden_states[0].size()
        
        # Merge all modalities and project
        all_q = torch.zeros(bsz, q_len, self.num_heads * self.head_dim, 
                           device=hidden_states[0].device, dtype=hidden_states[0].dtype)
        all_k = torch.zeros(bsz, q_len, self.num_key_value_heads * self.head_dim,
                           device=hidden_states[0].device, dtype=hidden_states[0].dtype)
        all_v = torch.zeros(bsz, q_len, self.num_key_value_heads * self.head_dim,
                           device=hidden_states[0].device, dtype=hidden_states[0].dtype)
        
        for i, mod_val_pos in enumerate(self.valid_pos):
            all_q[mod_val_pos] = self.q_proj[i](hidden_states[i][mod_val_pos])
            all_k[mod_val_pos] = self.k_proj[i](hidden_states[i][mod_val_pos])
            all_v[mod_val_pos] = self.v_proj[i](hidden_states[i][mod_val_pos])
        
        # Reshape for attention
        query_states = all_q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = all_k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = all_v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        
        # Project back to each modality's dimension
        final_outputs = []
        for i, mod_val_pos in enumerate(self.valid_pos):
            out = torch.zeros_like(hidden_states[i])
            out[mod_val_pos] = self.o_proj[i](attn_output[mod_val_pos])
            final_outputs.append(out)
        
        return final_outputs, None, past_key_value if use_cache else None
    
    def _forward_separate(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """Forward with separate attention for each modality."""
        final_outputs = []
        
        for i, hidden_state in enumerate(hidden_states):
            bsz, q_len, _ = hidden_state.size()
            
            query_states = self.q_proj[i](hidden_state)
            key_states = self.k_proj[i](hidden_state)
            value_states = self.v_proj[i](hidden_state)
            
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            # Apply RoPE
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # Repeat KV for GQA
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            # Attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj[i](attn_output)
            
            final_outputs.append(attn_output)
        
        return final_outputs, None, past_key_value if use_cache else None


class MoTLlamaDecoderLayer(nn.Module):
    """MoT LLaMA decoder layer with multi-modality support."""
    
    def __init__(self, config: LlamaConfig, layer_idx: int, modality_num: int = 2, mot_factor: float = 1.0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.modality_num = modality_num
        self.mot_factor = mot_factor
        
        self.hidden_size = config.hidden_size
        self.mot_embed_dim = int(config.hidden_size * mot_factor)
        self.embed_dims = [self.hidden_size, self.mot_embed_dim]
        
        # Determine if this layer uses shared attention
        self.shared_attn = layer_idx in getattr(config, 'cross_model_attention', list(range(config.num_hidden_layers)))
        
        # Self attention
        self.self_attn = MoTLlamaAttention(
            config=config,
            layer_idx=layer_idx,
            embed_dims=self.embed_dims,
            modality_num=modality_num,
            shared_attn=self.shared_attn,
        )
        
        # MLP for each modality
        intermediate_size = config.intermediate_size
        mot_intermediate_size = int(intermediate_size * mot_factor)
        
        mlps = [
            MoTLlamaMLP(self.hidden_size, intermediate_size),
            MoTLlamaMLP(self.mot_embed_dim, mot_intermediate_size),
        ]
        self.mlp = MoTDiffFuncMod(mlps, modality_num, out_dims=self.embed_dims)
        
        # Layer norms
        self.input_layernorm = MoTLlamaRMSNorm(
            [LlamaRMSNorm(dim, eps=config.rms_norm_eps) for dim in self.embed_dims],
            modality_num
        )
        self.post_attention_layernorm = MoTLlamaRMSNorm(
            [LlamaRMSNorm(dim, eps=config.rms_norm_eps) for dim in self.embed_dims],
            modality_num
        )
        
        self.valid_pos = None
        
    def update_typeids(self, valid_pos):
        self.valid_pos = valid_pos
        self.self_attn.update_typeids(valid_pos)
        self.input_layernorm.update_typeids(valid_pos)
        self.post_attention_layernorm.update_typeids(valid_pos)
        
    def forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        residual = hidden_states
        
        # Pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        # Residual connection
        hidden_states = [residual[i] + hidden_states[i] for i in range(self.modality_num)]
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP to each modality
        mlp_outputs = []
        for i, hid in enumerate(hidden_states):
            mlp_outputs.append(self.mlp.fn[i](hid))
        hidden_states = mlp_outputs
        
        # Residual connection
        hidden_states = [residual[i] + hidden_states[i] for i in range(self.modality_num)]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs


class MoTLlamaModel(nn.Module):
    """MoT LLaMA model backbone."""
    
    def __init__(self, config: LlamaConfig, modality_num: int = 2, mot_factor: float = 1.0):
        super().__init__()
        self.config = config
        self.modality_num = modality_num
        self.mot_factor = mot_factor
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.hidden_size = config.hidden_size
        self.mot_embed_dim = int(config.hidden_size * mot_factor)
        self.embed_dims = [self.hidden_size, self.mot_embed_dim]
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            MoTLlamaDecoderLayer(config, layer_idx, modality_num, mot_factor)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = MoTLlamaRMSNorm(
            [LlamaRMSNorm(dim, eps=config.rms_norm_eps) for dim in self.embed_dims],
            modality_num
        )
        
        self.valid_pos = None
        self.position_ids = None
        
    def update_typeids(self, valid_pos):
        self.valid_pos = valid_pos
        self.norm.update_typeids(valid_pos)
        for layer in self.layers:
            layer.update_typeids(valid_pos)
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            inputs_embeds = [self.embed_tokens(input_ids)]
        
        hidden_states = inputs_embeds
        batch_size, seq_length = hidden_states[0].shape[:2]
        
        # Position IDs
        if position_ids is None:
            device = hidden_states[0].device
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        
        # Causal mask
        if attention_mask is not None and attention_mask.dim() == 2:
            # Create 4D causal mask from 2D mask
            causal_mask = self._prepare_attention_mask(attention_mask, (batch_size, seq_length), hidden_states[0])
        else:
            causal_mask = attention_mask
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _prepare_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        """Prepare 4D causal attention mask."""
        batch_size, seq_length = input_shape
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), torch.finfo(dtype).min, device=device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        
        # Expand attention mask
        if attention_mask is not None:
            # [batch, seq] -> [batch, 1, 1, seq]
            expanded_mask = attention_mask[:, None, None, :].to(dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            causal_mask = causal_mask + expanded_mask
        
        return causal_mask
