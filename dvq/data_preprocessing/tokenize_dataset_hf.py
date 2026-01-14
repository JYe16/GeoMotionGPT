"""
Tokenize HumanML3D dataset using the pretrained DVQ from HuggingFace.

This script downloads the motion tokenizer from HuggingFace (zy22b/GeoMotionGPT)
and converts motion features to discrete tokens.

Usage:
    python dvq/data_preprocessing/tokenize_dataset_hf.py
    python dvq/data_preprocessing/tokenize_dataset_hf.py --data_root datasets/humanml3d/
"""

import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


def load_tokenizer_from_huggingface(repo_id="zy22b/GeoMotionGPT", device="cuda"):
    """
    Load the motion tokenizer from HuggingFace.
    
    Args:
        repo_id: HuggingFace repository ID
        device: Device to load the model on
        
    Returns:
        motion_tokenizer: The motion tokenizer model
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from dvq.softvq import VQVAE_148
    
    print(f"Loading motion tokenizer from HuggingFace ({repo_id})...")
    
    # Download model.safetensors from HuggingFace (contains both LM and tokenizer)
    model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    
    # Load full state dict
    full_state_dict = load_file(model_path)
    
    # Extract motion_tokenizer weights (remove "motion_tokenizer." prefix)
    tokenizer_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith('motion_tokenizer.'):
            new_key = k[len('motion_tokenizer.'):]
            tokenizer_state_dict[new_key] = v
    
    print(f"  Found {len(tokenizer_state_dict)} motion tokenizer weights")
    
    # Initialize DVQ model
    model = VQVAE_148(quantizer="gsst", nb_code=512, vec_size=263, down_t=3)
    
    # Load weights
    model.load_state_dict(tokenizer_state_dict)
    model.to(device)
    model.eval()
    
    print("✓ Motion tokenizer loaded successfully!")
    return model


def tokenize_motion(model, motion_data, mean, std, device="cuda"):
    """
    Tokenize a single motion sequence.
    
    Args:
        model: The DVQ model
        motion_data: Motion feature array of shape (T, 263)
        mean: Mean for z-score normalization
        std: Std for z-score normalization
        device: Device to run inference on
        
    Returns:
        tokens: Token indices of shape (T//4,)
    """
    with torch.no_grad():
        # Convert to tensor
        motion_tensor = torch.from_numpy(motion_data).float().to(device)
        
        # Z-Score Normalization (important!)
        motion_normalized = (motion_tensor - mean) / std
        
        # Add batch dimension
        motion_batch = motion_normalized.unsqueeze(0)
        
        # Encode to get tokens
        tokens = model.encode(motion_batch)
        
        # Remove batch dimension and convert to numpy
        tokens = tokens.squeeze(0).cpu().numpy()
    return tokens


def batch_tokenize(model, motion_vec_dir: str, token_dir: str, data_root: str, device="cuda"):
    """
    Batch process the entire dataset, converting motion feature vectors to token sequences.

    Args:
        model: The DVQ model
        motion_vec_dir (str): Input directory containing the (T, 263) feature vectors.
        token_dir (str): Output directory to save the (T_quantized,) token sequences.
        data_root (str): Root directory containing Mean.npy and Std.npy
        device: Device to run inference on
    """
    # Load mean and std for normalization
    mean_path = os.path.join(data_root, 'Mean.npy')
    std_path = os.path.join(data_root, 'Std.npy')
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print(f"Error: Mean.npy or Std.npy not found in {data_root}")
        print("Please ensure the HumanML3D dataset is properly preprocessed.")
        sys.exit(1)
    
    mean = torch.from_numpy(np.load(mean_path)).float().to(device)
    std = torch.from_numpy(np.load(std_path)).float().to(device)
    print(f"Loaded normalization stats from {data_root}")
    
    if not os.path.exists(token_dir):
        os.makedirs(token_dir)
        print(f"Created output directory: {token_dir}")

    files_to_process = [f for f in os.listdir(motion_vec_dir) if f.endswith('.npy')]

    print(f"Found {len(files_to_process)} motion files to tokenize...")
    success_count = 0
    error_count = 0

    for filename in tqdm(files_to_process, desc="Tokenizing"):
        input_path = os.path.join(motion_vec_dir, filename)
        output_path = os.path.join(token_dir, filename)

        try:
            # 1. Load the motion feature vector
            motion_data = np.load(input_path)

            # 2. Perform tokenization (with normalization)
            motion_tokens = tokenize_motion(model, motion_data, mean, std, device)

            # 3. Save the token sequence
            np.save(output_path, motion_tokens)
            success_count += 1

        except Exception as e:
            error_count += 1
            # Uncomment to debug errors:
            # print(f"Error processing {filename}: {e}")

    print(f"✓ Tokenization complete: {success_count} success, {error_count} errors")


def main():
    parser = argparse.ArgumentParser(description="Tokenize HumanML3D dataset using DVQ from HuggingFace")
    parser.add_argument('--data_root', type=str, default='datasets/humanml3d/',
                        help='Root directory of the HumanML3D dataset')
    parser.add_argument('--repo_id', type=str, default='zy22b/GeoMotionGPT',
                        help='HuggingFace repository ID')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Setup paths
    motion_vec_dir = os.path.join(args.data_root, 'new_joint_vecs')
    token_dir = os.path.join(args.data_root, 'motion_tokens')
    
    print("=" * 60)
    print("GeoMotionGPT - Motion Tokenization")
    print("=" * 60)
    print(f"HuggingFace repo: {args.repo_id}")
    print(f"Input directory:  {motion_vec_dir}")
    print(f"Output directory: {token_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Check input directory exists
    if not os.path.exists(motion_vec_dir):
        print(f"Error: Input directory not found: {motion_vec_dir}")
        print("Please download and preprocess the HumanML3D dataset first.")
        sys.exit(1)

    # Load tokenizer from HuggingFace
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print("Warning: CUDA not available, using CPU instead")
    
    model = load_tokenizer_from_huggingface(args.repo_id, device)

    # Run batch tokenization
    batch_tokenize(model, motion_vec_dir, token_dir, args.data_root, device)
    
    print(f"\n✓ Token sequences saved to: {token_dir}")


if __name__ == '__main__':
    main()