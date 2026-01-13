import numpy as np
import os
import argparse
import sys


def inspect_tokens(base_path, model_type, filename):
    result = f"Model type: {model_type}, File {filename}\n\n"

    for t in range(1, 6):
        file_path = os.path.join(base_path, model_type, f'run_{t}', filename)
        
        result += f"--- Run #{t} ---\n"
        
        if not os.path.exists(file_path):
            result += "File not found.\n"
            continue
            
        try:
            tokens = np.load(file_path)
            # Convert tokens to string
            np.set_printoptions(threshold=sys.maxsize)
            result += np.array2string(tokens, separator=', ') + "\n"
        except Exception as e:
            result += f"Error loading file: {e}\n"
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="../data/temp/consistency_analysis/")
    parser.add_argument('--type', type=str, default="gsst")
    parser.add_argument('--filename', type=str, default="000001.npy")
    args = parser.parse_args()
    
    result = inspect_tokens(args.base_path, args.type, args.filename)
    print(result)
