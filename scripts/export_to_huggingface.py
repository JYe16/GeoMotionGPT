"""
Export GeoMotionGPT to HuggingFace format with custom model code.

This script:
1. Loads the pretrained checkpoints (DVQ-GSST and fine-tuned GPT2)
2. Creates a GeoMotionGPTForCausalLM model
3. Transfers weights
4. Saves with push_to_hub() to include model code

Usage:
    python scripts/export_to_huggingface_v2.py --output_dir ./huggingface_export
    python scripts/export_to_huggingface_v2.py --push_to_hub --repo_id zy22b/GeoMotionGPT
"""

import os
import sys
import json
import argparse
import shutil

import torch
from safetensors.torch import save_file

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Need to temporarily modify the modeling file's imports
import importlib.util
hf_export_dir = os.path.join(project_root, 'huggingface_export')

# Load configuration module first
spec = importlib.util.spec_from_file_location("configuration_geomotiongpt", 
    os.path.join(hf_export_dir, "configuration_geomotiongpt.py"))
configuration_module = importlib.util.module_from_spec(spec)
sys.modules["configuration_geomotiongpt"] = configuration_module
spec.loader.exec_module(configuration_module)
GeoMotionGPTConfig = configuration_module.GeoMotionGPTConfig

# Patch the modeling file to use absolute import
modeling_path = os.path.join(hf_export_dir, "modeling_geomotiongpt.py")
with open(modeling_path, 'r') as f:
    modeling_code = f.read()

# Replace relative import with direct reference
modeling_code_patched = modeling_code.replace(
    "from .configuration_geomotiongpt import GeoMotionGPTConfig",
    "# Import patched for standalone execution\npass  # GeoMotionGPTConfig imported externally"
)

# Execute the patched code
exec(compile(modeling_code_patched, modeling_path, 'exec'), {
    '__name__': 'modeling_geomotiongpt',
    '__file__': modeling_path,
    'GeoMotionGPTConfig': GeoMotionGPTConfig,
    'torch': __import__('torch'),
    'nn': __import__('torch').nn,
    'F': __import__('torch').nn.functional,
    'Optional': __import__('typing').Optional,
    'Tuple': __import__('typing').Tuple,
    'List': __import__('typing').List,
    'Union': __import__('typing').Union,
    'PreTrainedModel': __import__('transformers').PreTrainedModel,
    'GPT2LMHeadModel': __import__('transformers').GPT2LMHeadModel,
    'GPT2Config': __import__('transformers').GPT2Config,
    'CausalLMOutputWithCrossAttentions': __import__('transformers').modeling_outputs.CausalLMOutputWithCrossAttentions,
})

# Get the model class from globals
GeoMotionGPTForCausalLM = None
# We need a different approach - just copy the model class definition


def load_motion_tokenizer_weights(dvq_path: str):
    """Load DVQ-GSST motion tokenizer weights."""
    print(f"Loading motion tokenizer from {dvq_path}")
    state_dict = torch.load(dvq_path, map_location='cpu')
    
    # Map from original keys to new model keys
    new_state_dict = {}
    for key, value in state_dict.items():
        # Original: vqvae.encoder.model.X -> motion_tokenizer.encoder.model.X
        if key.startswith('vqvae.'):
            new_key = 'motion_tokenizer.' + key[6:]  # Remove 'vqvae.'
            new_state_dict[new_key] = value
    
    print(f"  Loaded {len(new_state_dict)} motion tokenizer parameters")
    return new_state_dict


def load_language_model_weights(ckpt_path: str):
    """Load fine-tuned GPT2 language model weights."""
    print(f"Loading language model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    new_state_dict = {}
    
    # Map language model weights
    lm_prefix = "lm.language_model."
    for key, value in state_dict.items():
        if key.startswith(lm_prefix):
            # lm.language_model.transformer.X -> language_model.transformer.X
            new_key = 'language_model.' + key[len(lm_prefix):]
            new_state_dict[new_key] = value
    
    print(f"  Loaded {len(new_state_dict)} language model parameters")
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Export GeoMotionGPT to HuggingFace")
    parser.add_argument("--dvq_path", type=str, default="checkpoints/dvq-gsst.pt",
                        help="Path to DVQ-GSST checkpoint")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/GeoMotionGPT.ckpt",
                        help="Path to GeoMotionGPT checkpoint")
    parser.add_argument("--output_dir", type=str, default="./huggingface_export",
                        help="Output directory")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push to HuggingFace Hub")
    parser.add_argument("--repo_id", type=str, default="zy22b/GeoMotionGPT",
                        help="HuggingFace repo ID")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GeoMotionGPT Export (with model code)")
    print("=" * 60)
    
    # Create config
    config = GeoMotionGPTConfig(
        motion_vocab_size=512,
        motion_input_dim=263,
        motion_hidden_dim=512,
        motion_down_t=3,
        motion_depth=3,
        motion_dilation_growth_rate=3,
        text_vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        mot_factor=1.0,
        attention_mode="all",
        lambda_geo=0.01,
    )
    
    # Create model
    print("\n[1/4] Creating model...")
    model = GeoMotionGPTForCausalLM(config)
    print(f"  Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load motion tokenizer weights
    print("\n[2/4] Loading motion tokenizer weights...")
    motion_weights = load_motion_tokenizer_weights(args.dvq_path)
    
    # Load language model weights  
    print("\n[3/4] Loading language model weights...")
    lm_weights = load_language_model_weights(args.ckpt_path)
    
    # Merge and load weights
    print("\n[4/4] Merging weights into model...")
    all_weights = {}
    all_weights.update(motion_weights)
    all_weights.update(lm_weights)
    
    # Load with strict=False to allow missing motion_embed, motion_head, etc.
    missing, unexpected = model.load_state_dict(all_weights, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing examples: {missing[:5]}")
    
    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    
    # Save configuration
    config.save_pretrained(args.output_dir)
    
    # Save model weights as safetensors
    state_dict = model.state_dict()
    
    # Handle shared tensors (clone if necessary)
    processed_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            processed_dict[key] = value.contiguous().clone()
    
    save_file(processed_dict, os.path.join(args.output_dir, "model.safetensors"))
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nFiles in {args.output_dir}:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {f}: {size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"  {f}: {size / 1024:.1f} KB")
    
    # Push to hub if requested
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
        model.push_to_hub(args.repo_id)
        config.push_to_hub(args.repo_id)
        print("Done!")
    
    print("\n" + "-" * 60)
    print("To upload manually:")
    print(f"  huggingface-cli upload {args.repo_id} {args.output_dir}")
    print("\nTo load the model:")
    print('  from transformers import AutoModelForCausalLM')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{args.repo_id}", trust_remote_code=True)')
    print('  motion_tokenizer = model.motion_tokenizer')
    print("-" * 60)


if __name__ == "__main__":
    main()
