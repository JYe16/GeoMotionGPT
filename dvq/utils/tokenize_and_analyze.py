import torch
import argparse
import os
import sys
import numpy as np
import shutil

project_root = '../'
sys.path.append(project_root)
from utils.define_device import define_device
from utils.code_count import count_codes_dual, count_codes
from data_preprocessing.tokenize_dataset import tokenize
from utils.inspect_tokens import inspect_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="../data/humanml3d_263/")
    parser.add_argument('--type', type=str, default="gsst")
    parser.add_argument('--nb_code', type=int, default=512)
    parser.add_argument('--output_path', type=str, default="../data/temp/consistency_analysis/")
    args = parser.parse_args()

    ckpt_path = os.path.join('../vqvae_checkpoints', 'humanml3d_263_' + args.type + '.pt')
    device = define_device()
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    if isinstance(checkpoint, dict):
        # It's a state_dict, instantiate the model
        if '2cb' in args.type:
            from network.double_codebook_vq.double_codebook_vq import HumanVQVAE
            quantizer = args.type.replace('_2cb', '')
            model = HumanVQVAE(quantizer=quantizer, nb_code=args.nb_code, vec_size=263)
        elif 'st_share' in args.type:
            from network.st_share.st_share_vq import HumanVQVAE
            model = HumanVQVAE(quantizer=args.type, nb_code=args.nb_code, vec_size=263)
        elif 'vqvae' in args.type:
            from network.vqvae.vqvae import HumanVQVAE
            model = HumanVQVAE(quantizer='ema_reset', nb_code=args.nb_code, vec_size=263)
        else:
            from network.gsst.softvq import HumanVQVAE
            model = HumanVQVAE(quantizer=args.type, nb_code=args.nb_code, vec_size=263)
        
        model.load_state_dict(checkpoint)
    else:
        # It's a full model
        model = checkpoint

    model.to(device)
    model.eval()

    # 1. Generate tokens 5 times
    for t in range(1, 6, 1):
        print(f'Run #{t}:')
        output_path = os.path.join(args.output_path, args.type, 'run_' + str(t))
        os.makedirs(output_path, exist_ok=True)
        tokenize(model=model, data_root=args.data_root, vec_size=263, output_path=output_path)
        if args.type == 'gsst_2cb':
            count_codes_dual(data_path=output_path, nb_code=args.nb_code, out_path=os.path.join(args.output_path, 'run_' + str(t) + '.csv'))
        else:
            count_codes(data_path=output_path, nb_code=args.nb_code, out_path=os.path.join(args.output_path,  args.type, 'run_' + str(t) + '.csv'))

    # 2. Consistency Check
    print("\n" + "="*30)
    print("Starting Consistency Analysis...")
    print("="*30)

    base_run_dir = os.path.join(args.output_path, args.type, 'run_1')
    if not os.path.exists(base_run_dir):
        print(f"Error: Base run directory {base_run_dir} does not exist.")
        sys.exit(1)

    files = sorted([f for f in os.listdir(base_run_dir) if f.endswith('.npy')])
    total_files = len(files)
    consistent_count = 0
    inconsistent_files = []

    print(f"Checking {total_files} files across 5 runs...")

    for filename in files:
        # Load from run 1
        path1 = os.path.join(base_run_dir, filename)
        try:
            tokens1 = np.load(path1)
        except Exception as e:
            print(f"Error loading {path1}: {e}")
            inconsistent_files.append(filename)
            continue

        is_consistent = True
        for t in range(2, 6):
            run_dir = os.path.join(args.output_path, args.type, f'run_{t}')
            path_t = os.path.join(run_dir, filename)
            
            if not os.path.exists(path_t):
                print(f" [!] File missing in run_{t}: {filename}")
                is_consistent = False
                break
            
            try:
                tokens_t = np.load(path_t)
            except Exception as e:
                print(f" [!] Error loading {path_t}: {e}")
                is_consistent = False
                break

            # Check shape and content
            if tokens1.shape != tokens_t.shape:
                print(f" [!] Shape mismatch for {filename}: Run 1 {tokens1.shape} vs Run {t} {tokens_t.shape}")
                is_consistent = False
                break
            
            if not np.array_equal(tokens1, tokens_t):
                print(f" [!] Content mismatch for {filename} between Run 1 and Run {t}")
                # Optional: print first mismatch index
                # mismatch_indices = np.where(tokens1 != tokens_t)[0]
                # if len(mismatch_indices) > 0:
                #     idx = mismatch_indices[0]
                #     print(f"     First mismatch at index {idx}: {tokens1.flatten()[idx]} vs {tokens_t.flatten()[idx]}")
                is_consistent = False
                break
        
        if is_consistent:
            consistent_count += 1
        else:
            inconsistent_files.append(filename)

    print("\n" + "="*30)
    print("Consistency Analysis Results")
    print("="*30)
    print(f"Total Files: {total_files}")
    print(f"Consistent:  {consistent_count}")
    print(f"Inconsistent: {len(inconsistent_files)}")
    if total_files > 0:
        print(f"Consistency Rate: {consistent_count / total_files * 100:.2f}%")
    else:
        print("Consistency Rate: N/A (0 files)")
    
    if len(inconsistent_files) > 0:
        print(f"\nInconsistent files (first 10): {inconsistent_files[:10]}")

    # Save consistency results to file
    result_file_path = os.path.join(args.output_path, args.type, f'ConsistencyResults_{args.type}.txt')
    with open(result_file_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Consistency Analysis Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model Type: {args.type}\n")
        f.write(f"Total Files: {total_files}\n")
        f.write(f"Consistent:  {consistent_count}\n")
        f.write(f"Inconsistent: {len(inconsistent_files)}\n")
        if total_files > 0:
            f.write(f"Consistency Rate: {consistent_count / total_files * 100:.2f}%\n")
        else:
            f.write("Consistency Rate: N/A (0 files)\n")
        
        if len(inconsistent_files) > 0:
            print(f"\nInconsistent files: {inconsistent_files}\n")
        
        # Inspect tokens for selected files
        f.write("\n\n" + "="*50 + "\n")
        f.write("Token Inspection for Selected Files\n")
        f.write("="*50 + "\n\n")
        
        filenames = [f'{i:06d}.npy' for i in range(1, 13)]
        
        for filename in filenames:
            f.write(f"\n{'='*50}\n")
            f.write(f"Inspecting tokens for file: {filename}\n")
            f.write('='*50 + "\n")
            result = inspect_tokens(os.path.join(args.output_path, ''), args.type, filename)
            f.write(result)
            f.write("\n")
    
    print(f"\n✓ Consistency results saved to: {result_file_path}")
    
    # Clean up run folders
    print("\nCleaning up run folders...")
    for t in range(1, 6):
        run_dir = os.path.join(args.output_path, args.type, f'run_{t}')
        if os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
                print(f"✓ Deleted {run_dir}")
            except Exception as e:
                print(f"✗ Error deleting {run_dir}: {e}")
    
    print("\n✓ Cleanup complete!")