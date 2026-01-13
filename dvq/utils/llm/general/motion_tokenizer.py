import torch
import numpy as np
import argparse
import os
import sys

project_root = '../../../../'
sys.path.append(project_root)

from dvq.utils.define_device import define_device


class MotionTokenizer:
    """
    A motion tokenizer encapsulating a trained VQ-VAE model.
    """

    def __init__(self, model, data_root, vqvae_checkpoint):
        self.data_root = data_root
        self.device = define_device()

        # 1. Load model architecture
        self.model = model

        # 3. Set to evaluation mode
        self.model.eval()

        # 4. Load mean and standard deviation for normalization
        try:
            self.mean = torch.from_numpy(np.load(os.path.join(self.data_root, 'Mean.npy'))).float().to(self.device)
            self.std = torch.from_numpy(np.load(os.path.join(self.data_root, 'Std.npy'))).float().to(self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Mean.npy or Std.npy file not found in {self.data_root} directory.")


    def tokenize(self, motion_feature_vector: np.ndarray) -> np.ndarray:
        """
        Convert a single motion feature vector sequence into a discrete token sequence.

        Args:
            motion_feature_vector (np.ndarray): Motion feature vector with shape (T, 148).

        Returns:
            np.ndarray: 1D integer token sequence with shape (T_quantized,).
        """
        # Ensure input is torch.Tensor and on the correct device
        motion_tensor = torch.from_numpy(motion_feature_vector).float().to(self.device)

        # 1. Z-Score Normalization
        motion_normalized = (motion_tensor - self.mean) / self.std

        # 2. Add a batch dimension -> (1, T, 148)
        motion_batch = motion_normalized.unsqueeze(0)

        # 3. Use model's encode method for encoding/tokenization
        with torch.no_grad():
            # The model's encode method returns a (1, T_quantized) tensor
            tokens_tensor = self.model.encode(motion_batch)

        # 4. Remove batch dimension and convert back to NumPy array
        for i in range(len(tokens_tensor)):
            tokens_tensor[i] = tokens_tensor[i].squeeze(0).cpu().numpy()

        return np.array(tokens_tensor)

    def __call__(self, motion_feature_vector: np.ndarray) -> np.ndarray:
        """Allow the tokenizer instance to be called directly"""
        return self.tokenize(motion_feature_vector)


def main():
    """
    An example main program demonstrating how to use MotionTokenizer.
    """
    parser = argparse.ArgumentParser(description="Motion Tokenizer Demo")

    # --- Key Arguments ---
    parser.add_argument('--vqvae_checkpoint', type=str, default="../../../model/vqvae/humanml3d_263.pt")
    parser.add_argument('--data_root', type=str, default="../../../data/humanml3d_263/")
    parser.add_argument('--motion_file', type=str, default="000000.npy")
    # Note: quantizer related parameters (like mu) are usually not needed during inference as we don't update the codebook

    args = parser.parse_args()

    # 1. Initialize Tokenizer
    tokenizer = MotionTokenizer(args, args.data_root)

    # 2. Load a motion file
    try:
        motion_data = np.load(os.path.join(args.data_root, 'new_joint_vecs', args.motion_file))
        print(f"\nSuccessfully loaded motion file: {args.motion_file}")
        print(f"Original motion shape: {motion_data.shape}")
    except Exception as e:
        print(f"Error: Unable to load motion file {args.motion_file}: {e}")
        return

    # 3. Perform tokenization
    motion_tokens = tokenizer.tokenize(motion_data)

    # 4. Print results
    print(f"\n分词后的Token序列 (前20个): {motion_tokens[:20]}")
    print(f"Token序列总长度: {len(motion_tokens)}")
    print(f"Token的数据类型: {motion_tokens.dtype}")
    print("\n分词成功！")


if __name__ == '__main__':
    main()
