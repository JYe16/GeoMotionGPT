# analyze_single_codebook.py
# Usage example:
#   python analyze_single_codebook.py \
#       --file /mnt/data/code_counts_gsst.csv \
#       --name "GSST-1024" \
#       --out_png lorenz_single.png \
#       --out_log codebook_single_summary.csv

import argparse
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_codebook(df: pd.DataFrame, name: str):
    """Calculate health metrics for a single codebook. df must contain column: 'count' ('code_id' optional)."""
    if "count" not in df.columns:
        raise ValueError("Input dataframe must contain a 'count' column.")
    counts = df["count"].astype(float).values

    total = counts.sum()
    probs = counts / (total + 1e-12)

    # Entropy / Perplexity
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    perplexity = float(np.exp(entropy))

    # Utilization and Activity
    utilization_ratio = float(np.mean(counts > 0))
    mean_count = counts.mean()
    active_25 = float(np.mean(counts > 0.25 * mean_count))
    active_50 = float(np.mean(counts > 0.50 * mean_count))
    active_100 = float(np.mean(counts > 1.00 * mean_count))

    # Gini Coefficient (Imbalance)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * sorted_counts.sum() + 1e-12) - (n + 1) / n
    gini = float(gini)

    stats = {
        "Name": name,
        "Num codes": int(n),
        "Perplexity": perplexity,
        "Gini": gini,
        "Std (usage)": float(counts.std()),
        "Min usage": int(counts.min()),
        "Max usage": int(counts.max()),
        "Utilization (>0)": utilization_ratio,
        "≥25% mean (ratio)": active_25,
        "≥50% mean (ratio)": active_50,
        "≥100% mean (ratio)": active_100,
    }
    return stats, counts

def lorenz_curve(counts: np.ndarray):
    """返回 Lorenz 曲线坐标 (x, y)。"""
    counts = np.sort(counts.astype(float))
    cum_counts = np.cumsum(counts)
    total = cum_counts[-1] if cum_counts[-1] > 0 else 1.0
    y = np.concatenate([[0.0], cum_counts / total])
    x = np.linspace(0.0, 1.0, y.size)
    return x, y

def top_k_coverage(counts: np.ndarray, k_frac: float):
    """前 k_frac*N 个最高频 code 覆盖的使用占比。"""
    n = len(counts)
    k = max(1, int(round(k_frac * n)))
    top = np.sort(counts)[-k:]
    return float(top.sum() / (counts.sum() + 1e-12))

def parse_top_fracs(s: str):
    """解析形如 '0.1,0.2,0.3' 的字符串为浮点列表。"""
    return [float(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out_path", type=str, default="/data/jackieye/llm-sensing-working-dir/codebook_analysis/")
    ap.add_argument("--top_fracs", type=str, default="0.1,0.2,0.3",
                    help="头部覆盖率阈值，逗号分隔，例如 '0.1,0.2,0.3'")
    args = ap.parse_args()

    save_dir = str(os.path.join(args.out_path, args.name))
    os.makedirs(save_dir, exist_ok=True)

    # 读取 CSV
    df = pd.read_csv(args.file)

    # 计算指标
    stats, counts = analyze_codebook(df, args.name)

    # 头部覆盖率
    fracs = parse_top_fracs(args.top_fracs)
    for frac in fracs:
        stats[f"Top {int(frac*100)}% codes coverage"] = top_k_coverage(counts, frac)

    # 保存/打印指标
    print("\n=== Codebook Health Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    out_log = os.path.join(save_dir, f"{args.name}_summary.log")

    with open(out_log, "w") as f:
        f.write("=== Single Codebook Health Summary ===\n")
        for k, v in stats.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"\nSaved metrics to: {out_log}")

    # 绘制 Lorenz（单图、无自定义颜色/样式）
    x, y = lorenz_curve(counts)
    plt.figure(figsize=(7, 6))
    plt.plot(x, y, label=f"{args.name} (Gini={stats['Gini']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect equality")  # 45 度线
    plt.title("Lorenz Curve of Code Usage")
    plt.xlabel("Cumulative share of codes")
    plt.ylabel("Cumulative share of usage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(save_dir, f"lorenz_{args.name}.png")
    plt.savefig(out_png, dpi=200)
    print(f"Saved Lorenz figure to: {out_png}")

if __name__ == "__main__":
    main()
