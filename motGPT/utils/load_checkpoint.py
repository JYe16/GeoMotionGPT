import torch
import sys


def is_lora_checkpoint(state_dict):
    """Check if checkpoint was saved with LoRA/PEFT wrapper."""
    for key in state_dict.keys():
        if 'base_model.model' in key or 'lora_' in key:
            return True
    return False


def convert_lora_state_dict_to_merged(state_dict):
    """
    Convert LoRA checkpoint state_dict to merged format.
    This removes the PEFT wrapper prefix and merges LoRA weights into base weights.
    
    LoRA format: 
        lm.language_model.base_model.model.transformer.h.0.attn.c_attn.base_layer.weight
        lm.language_model.base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight
        lm.language_model.base_model.model.transformer.h.0.attn.c_attn.lora_B.default.weight
    
    Target format:
        lm.language_model.transformer.h.0.attn.c_attn.weight
    """
    from collections import OrderedDict
    
    # Collect LoRA weights
    lora_A = {}
    lora_B = {}
    base_weights = {}
    other_weights = OrderedDict()
    
    for key, value in state_dict.items():
        if 'lora_A' in key:
            # Extract base key: remove lora_A.default.weight and base_model.model
            base_key = key.replace('.lora_A.default.weight', '.weight')
            base_key = base_key.replace('.base_model.model.', '.')
            lora_A[base_key] = value
        elif 'lora_B' in key:
            base_key = key.replace('.lora_B.default.weight', '.weight')
            base_key = base_key.replace('.base_model.model.', '.')
            lora_B[base_key] = value
        elif '.base_layer.' in key:
            # Base weight for LoRA layers
            base_key = key.replace('.base_layer.', '.')
            base_key = base_key.replace('.base_model.model.', '.')
            base_weights[base_key] = value
        elif '.base_model.model.' in key:
            # Non-LoRA weights that still have PEFT wrapper
            new_key = key.replace('.base_model.model.', '.')
            other_weights[new_key] = value
        else:
            # Other weights without PEFT wrapper
            other_weights[key] = value
    
    # Merge LoRA weights: W_merged = W_base + (B @ A) * scaling
    # Note: Default scaling in PEFT is lora_alpha / r, but since we saved
    # during training, the scaling should already be applied
    merged_state_dict = OrderedDict()
    merged_state_dict.update(other_weights)
    
    # Add base weights (for LoRA layers)
    for key, base_value in base_weights.items():
        if key in lora_A and key in lora_B:
            # Merge: W = W_base + B @ A (assuming scaling is 1 or handled by LoRA config)
            # For inference, we just need base weights since we'll apply LoRA separately
            # Actually, for test.py we should just load and apply LoRA properly
            pass
        merged_state_dict[key] = base_value
    
    # For any remaining LoRA keys, we need the merged weights
    # But since test.py doesn't apply LoRA, we need to merge them
    for key in lora_A:
        if key in lora_B and key in base_weights:
            # W_merged = W_base + alpha/r * B @ A
            # Typical default: alpha=16, r=8 -> scaling = 2
            # But we'll use scaling=1 since the training might have different settings
            A = lora_A[key]  # (r, in_features)
            B = lora_B[key]  # (out_features, r)
            base = base_weights[key]
            # For Conv1D in GPT2, weight is (out_features, in_features)
            # LoRA: B @ A gives (out_features, in_features)
            scaling = 2.0  # Default: lora_alpha(16) / r(8) = 2
            try:
                lora_delta = B @ A * scaling
                merged_state_dict[key] = base + lora_delta
            except Exception as e:
                print(f"Warning: Could not merge LoRA for {key}: {e}")
                merged_state_dict[key] = base
    
    return merged_state_dict


def apply_lora_for_testing(cfg, model, logger=None):
    """Apply LoRA wrapper to model before loading LoRA checkpoint."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        if logger:
            logger.warning("PEFT not installed. Cannot load LoRA checkpoint.")
        return False
    
    # Get LoRA config from cfg - check multiple possible locations
    lora_cfg = None
    if hasattr(cfg, 'LORA') and cfg.LORA is not None:
        lora_cfg = cfg.LORA
    elif hasattr(cfg, 'model') and hasattr(cfg.model, 'params') and hasattr(cfg.model.params, 'LORA'):
        lora_cfg = cfg.model.params.LORA
    
    if lora_cfg is None:
        if logger:
            logger.warning("No LORA config found in cfg.LORA or cfg.model.params.LORA")
        return False
    
    # Check if LORA is enabled
    lora_enabled = getattr(lora_cfg, 'ENABLED', False)
    if not lora_enabled:
        if logger:
            logger.info("LORA.ENABLED is False, skipping LoRA wrapper")
        return False
    
    r = getattr(lora_cfg, 'r', 8)
    lora_alpha = getattr(lora_cfg, 'lora_alpha', 16)
    lora_dropout = getattr(lora_cfg, 'lora_dropout', 0.05)
    target_modules = getattr(lora_cfg, 'target_modules', ['c_attn', 'c_proj'])
    if hasattr(target_modules, '__iter__') and not isinstance(target_modules, str):
        target_modules = list(target_modules)
    train_embeddings = getattr(lora_cfg, 'train_embeddings', 'motion_only')
    
    if logger:
        logger.info(f"Applying LoRA wrapper for testing:")
        logger.info(f"  r={r}, alpha={lora_alpha}, targets={target_modules}")
    
    # Get GPT2 model
    gpt2 = model.lm.language_model
    
    # Determine modules_to_save based on train_embeddings setting
    modules_to_save = None
    if train_embeddings == 'all':
        modules_to_save = ['wte', 'lm_head']
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,  # Set to True for testing
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    
    # Apply LoRA
    model.lm.language_model = get_peft_model(gpt2, peft_config)
    
    if logger:
        logger.info("LoRA wrapper applied successfully")
    
    return True


def load_pretrained(cfg, model, logger=None, phase="train"):
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    # Handle nested state_dict (some checkpoints have double-nested structure)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and len(state_dict) == 1:
        if logger is not None:
            logger.info("Detected nested state_dict, unwrapping...")
        state_dict = state_dict["state_dict"]
    
    # Check if this is a LoRA checkpoint
    if is_lora_checkpoint(state_dict):
        if logger:
            logger.info("Detected LoRA checkpoint, applying LoRA wrapper...")
        
        # Try to apply LoRA wrapper to model first
        lora_applied = apply_lora_for_testing(cfg, model, logger)
        
        if not lora_applied:
            # If can't apply LoRA, try to merge weights
            if logger:
                logger.info("Could not apply LoRA, attempting to merge LoRA weights...")
            state_dict = convert_lora_state_dict_to_merged(state_dict)
    
    # Handle embedding size mismatch (model may have extended vocab for motion tokens)
    model_state = model.state_dict()
    for key in ['lm.language_model.transformer.wte.weight', 'lm.language_model.lm_head.weight']:
        if key in state_dict and key in model_state:
            ckpt_shape = state_dict[key].shape
            model_shape = model_state[key].shape
            if ckpt_shape != model_shape:
                if logger:
                    logger.info(f"Handling size mismatch for {key}: checkpoint {ckpt_shape} -> model {model_shape}")
                # Copy checkpoint weights into model's larger tensor
                new_weight = model_state[key].clone()
                min_size = min(ckpt_shape[0], model_shape[0])
                new_weight[:min_size] = state_dict[key][:min_size]
                state_dict[key] = new_weight
    
    model.load_state_dict(state_dict, strict=False)
    model.epoch = ckpt.get('epoch', -1)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE, weights_only=False,
                            map_location="cpu")['state_dict']
    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        # if 'skel_embedding' in k: continue
        # if 'final_layer' in k:continue
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v

    if hasattr(model, 'vae'):
        model.vae.load_state_dict(vae_dict, strict=True)
    else:
        model.motion_vae.load_state_dict(vae_dict, strict=True)
    
    return model
