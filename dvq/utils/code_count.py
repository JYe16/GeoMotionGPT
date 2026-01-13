import os
import sys

project_root = '../'
sys.path.append(project_root)
import numpy as np

def count_codes(data_path, nb_code=512, out_path="/data/jackieye/code_counts.csv", topk=5):
    counts = np.zeros(nb_code, dtype=np.int64)
    for fname in os.listdir(data_path):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(data_path, fname)
        codes = np.load(fpath)          # Read directly
        codes = codes.ravel()           # Flatten
        counts += np.bincount(codes, minlength=nb_code)
    total = counts.sum()

    # Print the top 20 most frequent tokens
    idx = np.argsort(-counts)[:topk]
    for i in idx:
        print(f"code {i}: {counts[i]} ({counts[i] / total:.2%})")

    # Save to CSV
    np.savetxt(out_path,
               np.column_stack([np.arange(nb_code), counts]),
               fmt="%d",
               delimiter=",",
               header="code_id,count",
               comments="")
    print(f"[OK] saved to {out_path}")
    return counts


def count_codes_dual(data_path, nb_code=512, out_path="/data/jackieye/code_counts.csv", topk=5):
    counts_spatial = np.zeros(nb_code, dtype=np.int64)
    counts_temporal = np.zeros(nb_code, dtype=np.int64)

    for fname in os.listdir(data_path):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(data_path, fname)
        codes = np.load(fpath)  # shape: (2, T)
        if codes.ndim != 2 or codes.shape[0] != 2:
            print(f"[WARN] Unexpected shape {codes.shape} in {fname}, skipped.")
            continue

        spatial_codes = codes[0].ravel()
        temporal_codes = codes[1].ravel()

        counts_spatial += np.bincount(spatial_codes, minlength=nb_code)
        counts_temporal += np.bincount(temporal_codes, minlength=nb_code)

    total_spatial = counts_spatial.sum()
    total_temporal = counts_temporal.sum()

    # 打印 TopK
    print("\n[Spatial Code TopK]")
    idx_s = np.argsort(-counts_spatial)[:topk]
    for i in idx_s:
        print(f"spatial code {i}: {counts_spatial[i]} ({counts_spatial[i] / total_spatial:.2%})")

    print("\n[Temporal Code TopK]")
    idx_t = np.argsort(-counts_temporal)[:topk]
    for i in idx_t:
        print(f"temporal code {i}: {counts_temporal[i]} ({counts_temporal[i] / total_temporal:.2%})")

    # 合并为一个CSV文件
    combined = np.column_stack([
        np.arange(nb_code),
        counts_spatial,
        counts_temporal
    ])

    header = "code_id,spatial,temporal"
    np.savetxt(out_path, combined, fmt="%d", delimiter=",", header=header, comments="")

    print(f"[OK] saved combined result to {out_path}")

    return counts_spatial, counts_temporal
if __name__ == '__main__':
    data_root = "../data/humanml3d_263/"
    nb_code = 512
    _ = count_codes_dual(data_root, nb_code, topk=nb_code)
