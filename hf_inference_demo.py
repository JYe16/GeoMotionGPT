"""
GeoMotionGPT Full Inference Demo

This script demonstrates full motion-to-text generation using
the HuggingFace motion tokenizer + project's LM module.

Usage:
    python scripts/full_inference_demo.py --motion_file datasets/humanml3d/new_joint_vecs/000000.npy
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="GeoMotionGPT Full Inference Demo")
    parser.add_argument("--motion_file", type=str, 
                        default="datasets/humanml3d/new_joint_vecs/000000.npy",
                        help="Path to HumanML3D motion .npy file")
    parser.add_argument("--mean_file", type=str, 
                        default="datasets/humanml3d/Mean.npy",
                        help="Path to Mean.npy for normalization")
    parser.add_argument("--std_file", type=str, 
                        default="datasets/humanml3d/Std.npy",
                        help="Path to Std.npy for normalization")
    parser.add_argument("--config", type=str, 
                        default="configs/test/m2t_o1e-2.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/GeoMotionGPT.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print("GeoMotionGPT Full Inference Demo")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Motion file: {args.motion_file}")
    
    # Step 1: Load motion tokenizer from HuggingFace
    print("\n[1/4] Loading motion tokenizer from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "zy22b/GeoMotionGPT",
        trust_remote_code=True
    )
    motion_tokenizer = hf_model.motion_tokenizer.to(device)
    motion_tokenizer.eval()
    del hf_model  # Free memory
    print("✓ Motion tokenizer loaded")
    
    # Step 2: Build and load Language Model
    print("\n[2/4] Building and loading language model...")
    cfg = OmegaConf.load(args.config)
    default_cfg = OmegaConf.load("configs/default.yaml")
    cfg = OmegaConf.merge(default_cfg, cfg)
    
    from motGPT.config import instantiate_from_config
    
    lm_cfg = OmegaConf.to_container(cfg.model.params.lm, resolve=True)
    # Set dummy values for VAE params since we use HF motion_tokenizer
    lm_cfg['params']['vae_latent_channels'] = 512
    lm_cfg['params']['vae_latent_size'] = None
    lm = instantiate_from_config(lm_cfg)
    
    # Load LM weights from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    lm_state = {k[3:]: v for k, v in ckpt["state_dict"].items() if k.startswith('lm.')}
    lm.load_state_dict(lm_state, strict=False)
    
    lm = lm.to(device)
    lm.eval()
    print("✓ Language model loaded")
    
    # Step 3: Load and process motion
    print("\n[3/4] Tokenizing motion...")
    motion = np.load(args.motion_file)
    mean = np.load(args.mean_file)
    std = np.load(args.std_file)
    motion_norm = (motion - mean) / std
    motion_tensor = torch.FloatTensor(motion_norm).unsqueeze(0).to(device)
    print(f"  Motion shape: {motion_tensor.shape}")
    
    with torch.no_grad():
        motion_tokens = motion_tokenizer.encode(motion_tensor)
    print(f"  Motion tokens: {motion_tokens.shape[1]} tokens")
    print(f"  Tokens: {motion_tokens[0].tolist()}")
    
    # Step 4: Generate text
    print("\n[4/4] Generating text...")
    with torch.no_grad():
        lengths = [motion_tokens.shape[1]]
        # Note: generate_conditional for m2t returns cleaned_text directly
        cleaned_texts = lm.generate_conditional(
            motion_tokens=motion_tokens,
            lengths=lengths,
            task="m2t",
            stage="test",  # To get cleaned text output
        )
    
    print(f"\n✓ Generated Text: {cleaned_texts[0] if cleaned_texts else 'N/A'}")
    
    # Ground truth
    motion_id = os.path.basename(args.motion_file).replace('.npy', '')
    text_file = args.motion_file.replace('new_joint_vecs', 'texts').replace('.npy', '.txt')
    if os.path.exists(text_file):
        print(f"\n[Ground Truth from {motion_id}.txt]:")
        with open(text_file, 'r') as f:
            for line in f.readlines()[:3]:
                print(f"  - {line.strip().split('#')[0]}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
