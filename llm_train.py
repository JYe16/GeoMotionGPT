"""
Training script with orthogonality regularization for motion token embeddings.

This script is identical to train.py but uses the MotGPTOrtho model variant
that adds orthogonality loss to encourage motion token embeddings to be
mutually orthogonal in the LLM's embedding space.

Key Features:
1. Loads pre-trained VQ-VAE codebook weights
2. Applies sparse projection to map codebook dimensions (512) to GPT2 hidden size (768)
3. Initializes motion token embeddings with projected codebook weights
4. Adds orthogonality loss during training

Usage:
    python train_ortho.py --cfg configs/m2t_custom_ortho.yaml

The orthogonality loss weight (lambda_ortho) can be configured in the YAML file
under model.params.lambda_ortho
"""

import os

# Fix for PyTorch 2.6+ checkpoint loading
# Must be set BEFORE importing torch
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import torch

# Fix for PyTorch 2.6+ checkpoint loading with OmegaConf
# Add safe globals for checkpoint deserialization
try:
    from omegaconf import ListConfig, DictConfig
    torch.serialization.add_safe_globals([ListConfig, DictConfig])
except Exception:
    pass

# Monkey-patch torch.load to use weights_only=False by default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from motGPT.callback import build_callbacks, LossCSVLogger
from motGPT.config import parse_args, instantiate_from_config
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

# PEFT LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def load_vqvae_codebook(codebook_path: str, logger=None) -> torch.Tensor:
    """
    Load VQ-VAE codebook weights from checkpoint.
    
    Args:
        codebook_path: Path to the codebook checkpoint file
        logger: Optional logger for printing info
        
    Returns:
        codebook_weight: Tensor of shape [num_codes, code_dim], e.g., [512, 512]
    """
    if logger:
        logger.info(f"Loading VQ-VAE codebook from: {codebook_path}")
    
    checkpoint = torch.load(codebook_path, map_location='cpu')
    
    # Extract codebook weights
    codebook_key = 'vqvae.quantizer.codebook.weight'
    if codebook_key in checkpoint:
        codebook_weight = checkpoint[codebook_key]
    else:
        raise KeyError(f"Codebook key '{codebook_key}' not found in checkpoint. "
                      f"Available keys: {list(checkpoint.keys())[:10]}...")
    
    if logger:
        logger.info(f"Codebook shape: {codebook_weight.shape}")  # Expected: [512, 512]
    
    return codebook_weight


def infer_dataset_name(cfg) -> str:
    """
    Infer dataset name from config.

    Priority:
    1) cfg.DATASET_NAME (if provided)
    2) cfg.DATASET.target (contains "kit" / "humanml")
    3) cfg.DATASET.HUMANML3D.ROOT path hint
    """
    dataset_name = str(cfg.get('DATASET_NAME', '')).strip().lower()
    if dataset_name:
        return dataset_name

    dataset_target = str(cfg.DATASET.get('target', '')).lower() if 'DATASET' in cfg else ''
    if 'kit' in dataset_target:
        return 'kit'
    if 'humanml' in dataset_target or 'humanml3d' in dataset_target:
        return 'humanml3d'

    try:
        data_root = str(cfg.DATASET.HUMANML3D.ROOT).lower()
        if 'kit' in data_root:
            return 'kit'
        if 'humanml' in data_root or 'humanml3d' in data_root:
            return 'humanml3d'
    except Exception:
        pass

    return 'humanml3d'


def resolve_codebook_path(cfg, logger=None) -> str:
    """
    Resolve VQ-VAE codebook checkpoint path.

    New naming convention (preferred):
        checkpoints/<dataset>-dvq-<quantizer>.pt
    """
    configured_path = str(cfg.get('CODEBOOK_PATH', '')).strip()
    dataset_name = infer_dataset_name(cfg)
    quantizer_name = str(cfg.get('DVQ_QUANTIZER', 'gsst')).strip()

    preferred_path = os.path.join('checkpoints', f'{dataset_name}-dvq-{quantizer_name}.pt')

    if configured_path and os.path.exists(configured_path):
        if logger:
            logger.info(f"Using configured CODEBOOK_PATH: {configured_path}")
        return configured_path

    if os.path.exists(preferred_path):
        if logger:
            logger.info(f"Using dataset-aware codebook path: {preferred_path}")
        return preferred_path

    legacy_path = os.path.join('checkpoints', 'dvq-gsst.pt')
    if os.path.exists(legacy_path):
        if logger:
            logger.warning(
                f"Preferred codebook not found ({preferred_path}); fallback to legacy path: {legacy_path}"
            )
        return legacy_path

    return configured_path or preferred_path


def create_sparse_projection(input_dim: int, output_dim: int, sparsity: float = 0.9, 
                             seed: int = 42) -> torch.Tensor:
    """
    Create a sparse random projection matrix.
    
    This projects vectors from input_dim to output_dim using a sparse random matrix.
    Sparse projection can help preserve distances while being computationally efficient.
    
    Args:
        input_dim: Input dimension (e.g., 512 for codebook)
        output_dim: Output dimension (e.g., 768 for GPT2)
        sparsity: Fraction of zero entries (default 0.9 means 90% sparse)
        seed: Random seed for reproducibility
        
    Returns:
        projection_matrix: Tensor of shape [input_dim, output_dim]
    """
    torch.manual_seed(seed)
    
    # Create sparse random projection matrix
    # Following the approach in sparse random projections literature
    s = 1.0 / (1.0 - sparsity)  # scaling factor
    
    # Initialize with zeros
    projection = torch.zeros(input_dim, output_dim)
    
    # Fill with sparse entries: +sqrt(s), 0, -sqrt(s) with probabilities 1/2s, 1-1/s, 1/2s
    prob_nonzero = 1.0 / s
    mask = torch.rand(input_dim, output_dim) < prob_nonzero
    
    # Randomly assign +1 or -1 to non-zero entries
    signs = torch.randint(0, 2, (input_dim, output_dim)) * 2 - 1  # -1 or +1
    projection[mask] = signs[mask].float() * (s ** 0.5)
    
    # Normalize columns to have unit norm (optional, for better embedding initialization)
    # projection = projection / (projection.norm(dim=0, keepdim=True) + 1e-8)
    
    # Scale to preserve expected norms
    projection = projection / (input_dim ** 0.5)
    
    return projection


def initialize_motion_token_embeddings(model, codebook_weight: torch.Tensor, 
                                       original_vocab_size: int = 50257,
                                       model_type: str = "gpt2",
                                       logger=None):
    """
    Initialize motion token embeddings using projected VQ-VAE codebook weights.
    
    Supports two architectures:
    1. MoT architecture (GPT2): Motion tokens have separate embedding layer (pre_processors[1])
    2. Standard architecture (LLaMA): Motion tokens are appended to text vocab in shared embedding
    
    Args:
        model: The MotGPTOrtho model
        codebook_weight: Tensor of shape [512, 512] from VQ-VAE
        original_vocab_size: Original vocab size (50257 for GPT2, 128256 for LLaMA)
        model_type: "gpt2" or "llama"
        logger: Optional logger
    """
    lm = model.lm.language_model
    num_codes, code_dim = codebook_weight.shape  # [512, 512]
    
    # Check if this is MoT architecture (has pre_processors)
    is_mot_architecture = hasattr(lm, 'pre_processors') and len(lm.pre_processors) > 1
    
    if is_mot_architecture:
        # =====================================================
        # MoT Architecture: Initialize pre_processors[1] (motion embedding)
        # =====================================================
        if logger:
            logger.info(f"=" * 60)
            logger.info(f"[MoT Architecture] Initializing motion embedding (pre_processors[1])")
        
        motion_embedding_layer = lm.pre_processors[1]  # motion_und_head
        motion_lm_head = lm.post_processors[1]  # motion_gen_head
        
        motion_vocab_size, mot_embed_dim = motion_embedding_layer.weight.shape
        
        if logger:
            logger.info(f"  Model type: {model_type}")
            logger.info(f"  VQ-VAE Codebook: {num_codes} codes × {code_dim} dim")
            logger.info(f"  Motion embedding: {motion_vocab_size} vocab × {mot_embed_dim} dim")
        
        # Create sparse projection to mot_embed_dim
        projection_matrix = create_sparse_projection(
            input_dim=code_dim, 
            output_dim=mot_embed_dim,
            sparsity=0.9,
            seed=42
        )
        projected_codebook = codebook_weight @ projection_matrix
        
        # Normalize
        with torch.no_grad():
            target_std = 0.02
            projected_norms = projected_codebook.norm(dim=1, keepdim=True)
            projected_codebook = projected_codebook / (projected_norms + 1e-8) * (target_std * (mot_embed_dim ** 0.5))
            
            num_to_init = min(num_codes, motion_vocab_size)
            motion_embedding_layer.weight[:num_to_init] = projected_codebook[:num_to_init].to(
                motion_embedding_layer.weight.device).to(motion_embedding_layer.weight.dtype)
            
            if motion_lm_head is not None:
                motion_lm_head.weight[:num_to_init] = projected_codebook[:num_to_init].to(
                    motion_lm_head.weight.device).to(motion_lm_head.weight.dtype)
        
        if logger:
            logger.info(f"  Initialized {num_to_init} motion embeddings")
            logger.info(f"=" * 60)
    else:
        # =====================================================
        # Standard Architecture: Initialize text embedding layer at motion token indices
        # Motion tokens are appended after original vocab
        # =====================================================
        if logger:
            logger.info(f"=" * 60)
            logger.info(f"[Standard Architecture] Initializing motion tokens in shared embedding")
        
        # Get the shared embedding layer
        if model_type == "gpt2":
            embedding_layer = lm.transformer.wte
            lm_head = lm.lm_head if hasattr(lm, 'lm_head') else None
        elif model_type == "llama":
            embedding_layer = lm.model.embed_tokens
            lm_head = lm.lm_head if hasattr(lm, 'lm_head') else None
        elif model_type == "qwen":
            embedding_layer = lm.model.embed_tokens
            lm_head = lm.lm_head if hasattr(lm, 'lm_head') else None
        else:
            if logger:
                logger.warning(f"Unknown model_type {model_type}")
            return
        
        total_vocab_size, hidden_dim = embedding_layer.weight.shape
        
        if logger:
            logger.info(f"  Model type: {model_type}")
            logger.info(f"  VQ-VAE Codebook: {num_codes} codes × {code_dim} dim")
            logger.info(f"  Shared embedding: {total_vocab_size} vocab × {hidden_dim} hidden")
            logger.info(f"  Original vocab size: {original_vocab_size}")
            logger.info(f"  Motion token indices: {original_vocab_size} to {original_vocab_size + num_codes - 1}")
        
        # Create sparse projection to hidden_dim
        # IMPORTANT: Do all computation in float32 to avoid numerical issues with float16
        projection_matrix = create_sparse_projection(
            input_dim=code_dim, 
            output_dim=hidden_dim,
            sparsity=0.9,
            seed=42
        ).float()  # Ensure float32
        
        codebook_float = codebook_weight.float()  # Ensure float32
        projected_codebook = codebook_float @ projection_matrix
        
        # Normalize to match existing embedding scale
        with torch.no_grad():
            existing_norms = embedding_layer.weight[:original_vocab_size].float().norm(dim=1)
            mean_norm = existing_norms.mean().item()
            
            if logger:
                logger.info(f"  Existing embedding norms: mean={mean_norm:.4f}")
            
            # Normalize in float32
            projected_norms = projected_codebook.norm(dim=1, keepdim=True)
            projected_codebook = projected_codebook / (projected_norms + 1e-8) * mean_norm
            
            # Check for NaN/Inf before assignment
            if torch.isnan(projected_codebook).any() or torch.isinf(projected_codebook).any():
                if logger:
                    logger.warning(f"  WARNING: projected_codebook has NaN/Inf! Using random init instead.")
                # Fall back to random initialization
                projected_codebook = torch.randn(num_codes, hidden_dim) * 0.02
            
            # Initialize motion tokens in shared embedding
            motion_start_idx = original_vocab_size
            motion_end_idx = original_vocab_size + num_codes
            
            # Convert to target dtype at the very end
            final_embeddings = projected_codebook.to(embedding_layer.weight.device).to(embedding_layer.weight.dtype)
            
            # Clamp to safe range for float16 (max ~65504)
            if embedding_layer.weight.dtype == torch.float16:
                final_embeddings = final_embeddings.clamp(-65000, 65000)
            
            embedding_layer.weight[motion_start_idx:motion_end_idx] = final_embeddings
            
            if logger:
                logger.info(f"  Final embedding dtype: {final_embeddings.dtype}")
                logger.info(f"  Final embedding has NaN: {torch.isnan(final_embeddings).any()}")
                logger.info(f"  Final embedding has Inf: {torch.isinf(final_embeddings).any()}")
            
            # Also init lm_head if not tied
            if lm_head is not None and lm_head.weight is not embedding_layer.weight:
                lm_head.weight[motion_start_idx:motion_end_idx] = final_embeddings
        
        if logger:
            logger.info(f"  Initialized {num_codes} motion token embeddings")
            logger.info(f"=" * 60)


def apply_lora_to_llm(model, lora_config: dict, original_vocab_size: int = 50257, 
                      model_type: str = "gpt2", logger=None):
    """
    Apply LoRA (Low-Rank Adaptation) to the LLM model inside MotGPT.
    
    This significantly reduces the number of trainable parameters while
    maintaining model performance. Importantly, this also ensures the 
    motion token embeddings remain trainable.
    
    Args:
        model: The MotGPTOrtho model containing self.lm.language_model
        lora_config: Dictionary with LoRA hyperparameters:
            - r: LoRA rank (default: 8)
            - lora_alpha: LoRA alpha scaling (default: 16)
            - lora_dropout: Dropout probability (default: 0.05)
            - target_modules: Which modules to apply LoRA to
            - train_embeddings: 'all' (full embedding), 'motion_only' (only motion tokens), or False
        original_vocab_size: Original vocab size (50257 for GPT2, 128256 for LLaMA)
        model_type: "gpt2" or "llama"
        logger: Optional logger
    """
    if not PEFT_AVAILABLE:
        if logger:
            logger.warning("PEFT not installed. Run: pip install peft")
            logger.warning("Continuing without LoRA...")
        return model
    
    # Get LoRA hyperparameters with defaults
    r = lora_config.get('r', 8)
    lora_alpha = lora_config.get('lora_alpha', 16)
    lora_dropout = lora_config.get('lora_dropout', 0.05)
    
    # Default target modules based on model type
    if model_type == "gpt2":
        default_target_modules = ['c_attn', 'c_proj']
    elif model_type == "llama":
        default_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif model_type == "qwen":
        default_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    else:
        default_target_modules = ['c_attn', 'c_proj']
    
    target_modules = lora_config.get('target_modules', default_target_modules)
    train_embeddings = lora_config.get('train_embeddings', 'motion_only')
    
    if logger:
        logger.info(f"=" * 60)
        logger.info(f"Applying LoRA to {model_type.upper()}")
        logger.info(f"  r (rank): {r}")
        logger.info(f"  lora_alpha: {lora_alpha}")
        logger.info(f"  lora_dropout: {lora_dropout}")
        logger.info(f"  target_modules: {target_modules}")
        logger.info(f"  train_embeddings: {train_embeddings}")
    
    # Count parameters before LoRA
    llm = model.lm.language_model
    total_params_before = sum(p.numel() for p in llm.parameters())
    trainable_params_before = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    
    # Determine modules_to_save based on train_embeddings setting and model type
    modules_to_save = None
    if train_embeddings == 'all':
        if model_type == "gpt2":
            modules_to_save = ['wte', 'lm_head']
        elif model_type in ["llama", "qwen"]:
            modules_to_save = ['embed_tokens', 'lm_head']
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    
    # Apply LoRA
    model.lm.language_model = get_peft_model(llm, peft_config)
    
    # If train_embeddings == 'motion_only', manually enable gradients for motion tokens only
    if train_embeddings == 'motion_only':
        if logger:
            logger.info(f"  Enabling gradients only for motion token embeddings (indices {original_vocab_size}+)")
        
        # Get the embedding layer - need to navigate through PEFT wrapper
        peft_model = model.lm.language_model
        base_model = peft_model.base_model.model
        
        # Get embedding and lm_head based on model type
        if model_type == "gpt2":
            wte = base_model.transformer.wte
            lm_head = base_model.lm_head
        elif model_type in ["llama", "qwen"]:
            wte = base_model.model.embed_tokens
            lm_head = base_model.lm_head
        else:
            wte = base_model.transformer.wte
            lm_head = base_model.lm_head
        
        # Make embedding layer require grad
        wte.weight.requires_grad = True
        if lm_head.weight is not wte.weight:
            lm_head.weight.requires_grad = True
        
        # Register hook to zero out gradients for non-motion tokens
        def create_embedding_grad_hook(vocab_size):
            def hook(grad):
                mask = torch.ones_like(grad)
                mask[:vocab_size] = 0
                return grad * mask
            return hook
        
        wte.weight.register_hook(create_embedding_grad_hook(original_vocab_size))
        if lm_head.weight is not wte.weight:
            lm_head.weight.register_hook(create_embedding_grad_hook(original_vocab_size))
        
        num_motion_tokens = wte.weight.shape[0] - original_vocab_size
        hidden_dim = wte.weight.shape[1]
        motion_params = num_motion_tokens * hidden_dim
        if logger:
            logger.info(f"  Motion token embeddings: {num_motion_tokens} tokens × {hidden_dim} dim = {motion_params:,} params")
    
    # Count parameters after LoRA
    trainable_params_after = sum(
        p.numel() for p in model.lm.language_model.parameters() if p.requires_grad
    )
    
    # Count LoRA params specifically
    lora_params = sum(
        p.numel() for name, p in model.lm.language_model.named_parameters() 
        if p.requires_grad and 'lora' in name.lower()
    )
    
    if logger:
        logger.info(f"  Total parameters: {total_params_before:,}")
        logger.info(f"  Trainable before LoRA: {trainable_params_before:,}")
        logger.info(f"  Trainable after LoRA (technical): {trainable_params_after:,}")
        logger.info(f"  LoRA adapter params: {lora_params:,}")
        
        if train_embeddings == 'motion_only':
            peft_model = model.lm.language_model
            base_model = peft_model.base_model.model
            
            if model_type == "gpt2":
                wte = base_model.transformer.wte
                lm_head = base_model.lm_head
            elif model_type in ["llama", "qwen"]:
                wte = base_model.model.embed_tokens
                lm_head = base_model.lm_head
            else:
                wte = base_model.transformer.wte
                lm_head = base_model.lm_head
                
            total_vocab = wte.weight.shape[0]
            hidden_dim = wte.weight.shape[1]
            num_motion = total_vocab - original_vocab_size
            
            wte_tied = wte.weight is lm_head.weight
            multiplier = 1 if wte_tied else 2
            
            motion_emb_params = num_motion * hidden_dim * multiplier
            effective_trainable = lora_params + motion_emb_params
            
            logger.info(f"  Motion embedding params: {motion_emb_params:,} ({num_motion} tokens × {hidden_dim} dim" + 
                       (f", tied)" if wte_tied else f" × 2)"))
            logger.info(f"  Effective trainable: {effective_trainable:,}")
        
        reduction = (1 - trainable_params_after / trainable_params_before) * 100
        logger.info(f"  Technical parameter reduction: {reduction:.1f}%")
        logger.info(f"=" * 60)
    
    # Print trainable parameters summary
    if logger:
        model.lm.language_model.print_trainable_parameters()
    
    return model


# Keep the old function name for backward compatibility
def apply_lora_to_gpt2(model, lora_config: dict, original_vocab_size: int = 50257, logger=None):
    """Backward compatible wrapper for apply_lora_to_llm with GPT2."""
    return apply_lora_to_llm(model, lora_config, original_vocab_size, "gpt2", logger)


def freeze_non_motion_embeddings(model, original_vocab_size: int, model_type: str, logger=None):
    """
    Freeze the non-motion token embeddings.
    
    For MoT architecture (GPT2): Freeze text embedding (pre_processors[0])
    For Standard architecture (LLaMA): Freeze original vocab in shared embedding using gradient hook
    
    Args:
        model: The MotGPTOrtho model (possibly with LoRA wrapper)
        original_vocab_size: Original vocab size before adding motion tokens
        model_type: "gpt2" or "llama"
        logger: Optional logger
    """
    llm = model.lm.language_model
    
    # Navigate through PEFT wrapper
    if hasattr(llm, 'base_model') and hasattr(llm.base_model, 'model'):
        base_model = llm.base_model.model
    else:
        base_model = llm
    
    # Check if MoT architecture
    is_mot_architecture = hasattr(base_model, 'pre_processors') and len(base_model.pre_processors) > 1
    
    if is_mot_architecture:
        # =====================================================
        # MoT Architecture: Freeze text embedding (pre_processors[0])
        # Motion embedding (pre_processors[1]) remains trainable
        # =====================================================
        if logger:
            logger.info(f"=" * 60)
            logger.info(f"[MoT] Freezing text embedding (pre_processors[0])")
        
        text_embed = base_model.pre_processors[0]
        text_lm_head = base_model.post_processors[0] if hasattr(base_model, 'post_processors') else None
        
        # Freeze text embedding
        text_embed.weight.requires_grad = False
        if text_lm_head is not None and text_lm_head.weight is not text_embed.weight:
            text_lm_head.weight.requires_grad = False
        
        # Ensure motion embedding is trainable
        motion_embed = base_model.pre_processors[1]
        motion_lm_head = base_model.post_processors[1] if len(base_model.post_processors) > 1 else None
        motion_embed.weight.requires_grad = True
        if motion_lm_head is not None:
            motion_lm_head.weight.requires_grad = True
        
        if logger:
            logger.info(f"  Text embedding frozen: {text_embed.weight.shape}")
            logger.info(f"  Motion embedding trainable: {motion_embed.weight.shape}")
            logger.info(f"=" * 60)
    else:
        # =====================================================
        # Standard Architecture: Use gradient hook to freeze original vocab
        # =====================================================
        if logger:
            logger.info(f"=" * 60)
            logger.info(f"[Standard] Freezing original vocab in shared embedding")
        
        # Get shared embedding layer
        if model_type == "gpt2":
            embed_layer = base_model.transformer.wte
            lm_head = base_model.lm_head if hasattr(base_model, 'lm_head') else None
        elif model_type in ["llama", "qwen"]:
            embed_layer = base_model.model.embed_tokens
            lm_head = base_model.lm_head if hasattr(base_model, 'lm_head') else None
        else:
            if logger:
                logger.warning(f"Unknown model_type {model_type}")
            return
        
        total_vocab_size = embed_layer.weight.shape[0]
        hidden_dim = embed_layer.weight.shape[1]
        num_motion_tokens = total_vocab_size - original_vocab_size
        
        if logger:
            logger.info(f"  Total vocab: {total_vocab_size}")
            logger.info(f"  Original vocab (frozen): {original_vocab_size}")
            logger.info(f"  Motion tokens (trainable): {num_motion_tokens}")
        
        # Gradient hook to zero out gradients for original vocabulary
        def create_freeze_hook(frozen_size):
            def hook(grad):
                mask = torch.ones_like(grad)
                mask[:frozen_size] = 0
                return grad * mask
            return hook
        
        embed_layer.weight.requires_grad = True
        embed_layer.weight.register_hook(create_freeze_hook(original_vocab_size))
        
        if lm_head is not None and lm_head.weight is not embed_layer.weight:
            lm_head.weight.requires_grad = True
            lm_head.weight.register_hook(create_freeze_hook(original_vocab_size))
        
        if logger:
            logger.info(f"  Registered gradient hook to freeze first {original_vocab_size} tokens")
            logger.info(f"=" * 60)


def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Log orthogonality loss info if using MotGPTOrtho
    if 'ortho' in cfg.model.target.lower():
        lambda_ortho = cfg.model.params.get('lambda_ortho', 0.1)
        logger.info(f"=" * 50)
        logger.info(f"Using MotGPTOrtho with orthogonality regularization")
        logger.info(f"lambda_ortho = {lambda_ortho}")
        logger.info(f"=" * 50)

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tenosrboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    
    # Add LossCSVLogger callback to record epoch losses
    loss_csv_logger = LossCSVLogger(
        output_dir=cfg.FOLDER_EXP,
        filename="epoch_losses.csv"
    )
    callbacks.append(loss_csv_logger)
    logger.info(f"LossCSVLogger will save to: {cfg.FOLDER_EXP}/epoch_losses.csv")
    
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # =====================================================
    # NEW: Initialize motion token embeddings from VQ-VAE codebook
    # =====================================================
    codebook_init_enabled = cfg.get('CODEBOOK_INIT_ENABLED', True)  # Default: enabled
    dataset_name = infer_dataset_name(cfg)
    codebook_path = resolve_codebook_path(cfg, logger)

    logger.info(f"Detected dataset: {dataset_name}")
    
    # Determine model type and original vocab size
    model_type = cfg.model.params.lm.params.get('model_type', 'gpt2')
    if model_type == 'gpt2':
        original_vocab_size = 50257
    elif model_type == 'llama':
        original_vocab_size = 128256  # LLaMA 3.2 vocab size
    elif model_type == 'qwen':
        original_vocab_size = 151936  # Qwen3 vocab size
    else:
        original_vocab_size = 50257  # Default to GPT2
    
    if not codebook_init_enabled:
        logger.info(f"="*60)
        logger.info(f"CODEBOOK_INIT_ENABLED=False: Skipping codebook initialization")
        logger.info(f"Using default random initialization for motion token embeddings")
        logger.info(f"="*60)
    elif os.path.exists(codebook_path):
        try:
            # Load VQ-VAE codebook weights
            codebook_weight = load_vqvae_codebook(codebook_path, logger)
            
            # Initialize motion token embeddings with sparse projection
            initialize_motion_token_embeddings(
                model=model,
                codebook_weight=codebook_weight,
                original_vocab_size=original_vocab_size,
                model_type=model_type,
                logger=logger
            )
        except Exception as e:
            logger.warning(f"Failed to initialize motion embeddings from codebook: {e}")
            logger.warning("Continuing with random initialization...")
    else:
        logger.warning(f"Codebook file not found at: {codebook_path}")
        logger.warning(f"Expected new naming: checkpoints/{dataset_name}-dvq-<quantizer>.pt")
        logger.warning("Using default random initialization for motion token embeddings")

    # =====================================================
    # NEW: Apply LoRA to LLM for parameter-efficient training
    # =====================================================
    lora_config = cfg.get('LORA', None)
    if lora_config and lora_config.get('ENABLED', False):
        model = apply_lora_to_llm(
            model=model,
            lora_config=lora_config,
            original_vocab_size=original_vocab_size,
            model_type=model_type,
            logger=logger
        )
    else:
        logger.info(f"LoRA disabled. Training full {model_type} model.")

    # =====================================================
    # NEW: Freeze non-motion token embeddings
    # - MoT architecture (GPT2): Freeze text embedding entirely
    # - Standard architecture (LLaMA): Freeze original vocab via gradient hook
    # =====================================================
    freeze_non_motion_embeddings(
        model=model,
        original_vocab_size=original_vocab_size,
        model_type=model_type,
        logger=logger
    )

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Determine precision based on model type
    model_type = cfg.model.params.lm.params.get('model_type', 'gpt2')
    if model_type in ['llama', 'qwen']:
        # Use float32 with gradient clipping for LLaMA/Qwen training stability
        train_precision = 32
        gradient_clip = 1.0  # Clip gradients to prevent explosion
    else:
        train_precision = 32
        gradient_clip = None

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
        accumulate_grad_batches=cfg.TRAIN.accumulate_grad_batches,
        precision=train_precision,
        gradient_clip_val=gradient_clip,
    )
    logger.info(f"Trainer initialized with precision={train_precision}, gradient_clip={gradient_clip}")

    # Strict load pretrained model (for fine-tuning, not resuming)
    if cfg.TRAIN.PRETRAINED and not cfg.TRAIN.RESUME:
        load_pretrained(cfg, model, logger)

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # Lightning Fitting
    # RESUME: Continue training from a checkpoint (restores model, optimizer, scheduler, epoch, etc.)
    # PRETRAINED: Only load model weights (for fine-tuning)
    # Note: resume_config() in config.py sets cfg.TRAIN.PRETRAINED to the actual checkpoint file path
    if cfg.TRAIN.RESUME:
        # Use cfg.TRAIN.PRETRAINED which was set by resume_config() to the actual .ckpt file
        resume_path = cfg.TRAIN.PRETRAINED
        logger.info(f"=" * 60)
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        logger.info(f"=" * 60)
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=resume_path)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
