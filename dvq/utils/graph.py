import os
import matplotlib.pyplot as plt

def update_plot(save_dir, ratio, lr, quantizer, 
                hist_epoch, hist_train_rec, hist_val_rec, 
                hist_train_util, hist_val_util, 
                fig, ax1, ax2, interactive_backend, 
                blocking_pause=False):
    """Update the realtime figure for Rec Loss (left y-axis) and Utilization (right y-axis)."""
    ax1.cla()
    ax2.cla()

    # Plot Rec Loss (left y-axis)
    ax1.plot(hist_epoch, hist_train_rec, label="Train Rec", color="tab:blue", linewidth=1.8)
    ax1.plot(hist_epoch, hist_val_rec, label="Val Rec", color="tab:orange", linewidth=1.8, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss")
    ax1.grid(True, alpha=0.3)

    # Plot Utilization (right y-axis)
    ax2.plot(hist_epoch, hist_train_util, label="Train Utilization", color="tab:green", linewidth=1.8)
    ax2.plot(hist_epoch, hist_val_util, label="Val Utilization", color="tab:red", linewidth=1.8,
             linestyle="--")
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Utilization")

    # Build a combined legend
    lines_left, labels_left = ax1.get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    ax1.legend(lines_left + lines_right, labels_left + labels_right, 
               loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)

    fig.tight_layout()

    # Interactive refresh if GUI backend is available
    if interactive_backend:
        plt.pause(0.001 if not blocking_pause else 0.05)

    # Always save a snapshot (useful on headless servers)
    img_filename = "training_curve_lr_" + str(lr) + "_ratio_" + str(ratio) + "_quantizer_" + quantizer + ".png"
    out_png = os.path.join(save_dir, img_filename)
    fig.savefig(out_png, dpi=150)
