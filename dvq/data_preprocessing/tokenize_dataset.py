import argparse
import os
import sys
from tqdm import tqdm
import numpy as np

project_root = '../../'
sys.path.append(project_root)

# --- Import Your Custom Modules ---
from dvq.utils.llm.general.motion_tokenizer import MotionTokenizer


def batch_tokenize(tokenizer: MotionTokenizer, motion_vec_dir: str, token_dir: str):
    """
    Batch process the entire dataset, converting motion feature vectors to token sequences.

    Args:
        tokenizer (MotionTokenizer): An initialized instance of the motion tokenizer.
        motion_vec_dir (str): Input directory containing the (T, 263) feature vectors.
        token_dir (str): Output directory to save the (T_quantized,) token sequences.
    """
    if not os.path.exists(token_dir):
        os.makedirs(token_dir)
        print(f"Created output directory: {token_dir}")

    files_to_process = [f for f in os.listdir(motion_vec_dir) if f.endswith('.npy')]

    print(f"Found {len(files_to_process)} motion files to tokenize...")
    c = 0

    for filename in tqdm(files_to_process, desc="Batch Tokenizing"):
        input_path = os.path.join(motion_vec_dir, filename)
        output_path = os.path.join(token_dir, filename)

        try:
            # 1. Load the motion feature vector
            motion_data = np.load(input_path)

            # 2. Perform tokenization
            motion_tokens = tokenizer.tokenize(motion_data)

            # 3. Save the token sequence
            np.save(output_path, motion_tokens)

        except Exception as e:
            c += 1
            # print(f"An error occurred while processing {filename}: {e}")


def tokenize(data_root, model, vec_size, output_path):
    """
    Main execution function
    """
    # 1. Initialize the tokenizer
    # We only need to initialize it once and then reuse it for all files for efficiency.
    tokenizer = MotionTokenizer(model, data_root, vec_size)

    # 2. Execute the batch processing
    batch_tokenize(tokenizer, os.path.join(data_root, 'new_joint_vecs'), output_path)
    print(f"Token sequences have been saved to: {data_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../dataset/humanml3d/')
    parser.add_argument('--vec_size', type=int, default=263)
    parser.add_argument('--vqvae_checkpoint', type=str, default="../vqvae_checkpoints/humanml3d_263_gsst.pt")
    parser.add_argument('--quantizer', type=str, default="gsst")
    parser.add_argument('--nb_code', type=int, default=512)
    args = parser.parse_args()

    print(f"Input directory (feature vectors): {os.path.join(args.data_root, 'new_joint_vecs')}")
    print(f"Output directory (tokens): {os.path.join(args.data_root, 'motion_tokens')}")
    
    tokenize(data_root=args.data_root, vec_size=args.vec_size, down_t=3, vqvae_checkpoint=args.vqvae_checkpoint, quantizer=args.quantizer, nb_code=args.nb_code)