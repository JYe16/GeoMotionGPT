import argparse
import os
import shutil
import sys
from datetime import datetime

project_root = '../'
sys.path.append(project_root)
from dvq.utils.define_working_dir import define_working_dir
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dvq.utils.define_device import define_device
from dvq.utils.training_related import save_to_log
from dvq.data_preprocessing.tokenize_dataset import tokenize as tokenize_test
from dvq.utils.code_count import count_codes
from dvq.utils.graph import update_plot
import matplotlib
import os  # if not already imported at top

# Use non-GUI backend if DISPLAY is not available (e.g., on servers)
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

# try:
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)
# except NameError:
#     pass

from dvq.softvq import HumanVQVAE
# Use the previously written data loader adapted for SMPL data
from dvq.dataloader.humanml3d.humanml3d_263_dataset_mgpt import HumanML3DDataset
from dvq.utils.define_num_workers import define_num_workers


def calculate_loss_ortho(model):
    C = model.vqvae.quantizer.codebook.weight
    C_norm = F.normalize(C, dim=-1)  # L2 normalization for each vector
    G = C_norm @ C_norm.t()  # [K, K] Gram matrix
    I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    loss_ortho = F.mse_loss(G, I)
    return loss_ortho


def set_quantizer_requires_grad(flag: bool, quantizer_params):
    for p in quantizer_params:
        p.requires_grad = flag

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="VQ-VAE Model Training Script")
    parser.add_argument('--data_root', type=str, default="datasets/humanml3d/")
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--ratio', type=float, default=4.0)
    parser.add_argument('--self_entropy_ratio', type=float, default=0.0)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--nb_code', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=define_num_workers())
    parser.add_argument('--quantizer', type=str, default="gsst")


    args = parser.parse_args()

    # --- 2. Setup ---
    # Create save directory
    now = datetime.now()
    working_dir = os.path.join('experiments', 'dvq',str(now.strftime("%m%d%Y") + '-' + args.quantizer + '-r' + str(args.ratio)) + '-c' + str(
                                   args.nb_code))
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Set device (GPU or CPU)
    device = define_device()
    save_to_log(f"Using device: {device}", working_dir=working_dir, print_msg=True)
    save_to_log(f"Codebook Size: {args.nb_code}", working_dir=working_dir, print_msg=True)

    # --- 3. Data Loading ---
    save_to_log("Loading dataset...", working_dir=working_dir, print_msg=True)
    # train_set = NTU60SklDataset(data_root=args.data_root, split="train", window_size=args.window_size)
    train_set = HumanML3DDataset(data_root=args.data_root, split="train", window_size=args.window_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)

    # val_set = NTU60SklDataset(data_root=args.data_root, split="val", window_size=args.window_size)
    val_set = HumanML3DDataset(data_root=args.data_root, split="val", window_size=args.window_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    # --- 4. Model & Optimizer Initialization ---
    # !!! IMPORTANT: Ensure parameters here match your VQVAE class definition
    # quantizer choices=['ema_reset', 'orig', 'ema', 'reset']
    model = HumanVQVAE(
        nb_code=args.nb_code,
        vec_size=263,
        quantizer=args.quantizer,
        down_t=3,
        ratio=args.ratio,
        self_entropy_ratio=args.self_entropy_ratio
    ).to(device)

    # --- build param groups: encoder/decoder vs. quantizer ---
    with_decay, no_decay, quantizer_all = [], [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # put all quantizer params (codebook, logit_scale, etc.) into their own group
        if "vqvae.quantizer" in n or "quantizer" in n:
            quantizer_all.append(p)
        else:
            # standard rule: no decay for bias and norm params
            if n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
                no_decay.append(p)
            else:
                with_decay.append(p)

    optimizer = optim.AdamW(
        [
            {"params": with_decay, "lr": args.lr, "weight_decay": 1e-4},
            {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
            {"params": quantizer_all, "lr": args.lr * 0.5, "weight_decay": 0.0},
        ]
    )

    # Scheduler: Cosine Annealing with Warmup
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.03 * total_steps)  # Can be changed to other ratios
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # --- 5. Training & Validation Loop ---
    save_to_log("Starting training loop...", working_dir=working_dir, print_msg=True)

    # --- Initialize realtime plot ---
    plt.ion()
    interactive_backend = (matplotlib.get_backend().lower() != "agg")

    hist_epoch = []
    hist_train_rec = []
    hist_val_rec = []
    hist_train_ppl = []
    hist_val_ppl = []

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    best_metric = float("inf")
    best_epoch = -1
    best_state = None  # will hold a CPU copy of the best state_dict
    # Update tau scheduler
    tau_start = 0.4
    tau_end = 0.01

    for epoch in range(args.num_epochs):
        # --- Training ---
        model.train()
        
        train_loss_rec = 0.0
        train_loss_util = 0.0
        train_loss_self_entropy = 0.0
        train_loss_ortho = 0.0
        train_util = 0.0

        # Update hard_ppl_rate
        if epoch < 150:
            hard_ppl_rate = 0.0
        elif epoch < 200:
            hard_ppl_rate = (epoch - 150) / 50.0
        else:
            hard_ppl_rate = 1.0

        # Update tau
        if epoch < 300:
            current_tau = tau_start
        else:
            current_tau = tau_start * (tau_end / tau_start) ** ((epoch - 300) / 100.0)
        
        model.vqvae.quantizer.tau = max(current_tau, tau_end)
        model.vqvae.quantizer.hard_ppl_rate = hard_ppl_rate
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for i, batch in enumerate(progress_bar):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            x_hat, loss_util, loss_self_entropy, util = model(batch)

            # Reconstruction Loss
            loss_rec = F.mse_loss(x_hat, batch)
            loss_ortho = calculate_loss_ortho(model)

            # Instead of ramping up to 1.0, let's keep it at a smaller, constant value
            # after the warm-up to prevent it from overwhelming the reconstruction loss.
            # w_vq = 0.1  # A common value used in many VQ-VAE papers.
            # Reduced loss_util weight from 1.0 to 0.02 because we have reset_dead_codes handling utilization now.
            loss = loss_rec + 0.25 * loss_util + loss_self_entropy + loss_ortho

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Record statistics
            train_loss_rec += loss_rec.item()
            train_loss_util += loss_util.item()
            train_loss_self_entropy += loss_self_entropy.item()
            train_loss_ortho += loss_ortho.item()
            train_util += util.item()

            progress_bar.set_postfix({
                # 'rec_loss': f'{loss_rec.item():.4f}',
                # 'util loss': f'{loss_util_item:.4f}',
                # 'self entropy loss': f'{loss_self_entropy_item:.4f}',
                # 'ortho loss': f'{loss_ortho.item():.4f}',
                # 'util': f'{util.item():.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                # 'soft rate': f'{hard_ppl_rate:.2f}',
                'tau': f'{model.vqvae.quantizer.tau:.4f}'
            })

        avg_train_loss_rec = train_loss_rec / len(train_loader)
        avg_train_loss_util = train_loss_util / len(train_loader)
        avg_train_loss_self_entropy = train_loss_self_entropy / len(train_loader)
        avg_train_loss_ortho = train_loss_ortho / len(train_loader)
        avg_train_util = train_util / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_rec = 0.0
        val_loss_util = 0.0
        val_loss_self_entropy = 0.0
        val_util = 0.0
        val_loss_ortho = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, loss_util, loss_self_entropy, util = model(batch)
                loss_rec = F.mse_loss(x_hat, batch)
                loss_ortho = calculate_loss_ortho(model)

                val_loss_rec += loss_rec.item()
                val_loss_util += loss_util.item()
                val_loss_self_entropy += loss_self_entropy.item()
                val_util += util.item()
                val_loss_ortho += loss_ortho.item()

        avg_val_loss_rec = val_loss_rec / len(val_loader)
        avg_val_loss_util = val_loss_util / len(val_loader)
        avg_val_loss_self_entropy = val_loss_self_entropy / len(val_loader)
        avg_val_util = val_util / len(val_loader)
        avg_val_loss_ortho = val_loss_ortho / len(val_loader)   
        val_total_loss = avg_val_loss_rec + 0.25 * avg_val_loss_util + avg_val_loss_self_entropy + avg_val_loss_ortho
        if val_total_loss < best_metric:
            best_metric = val_total_loss
            best_epoch = epoch + 1
            # keep a CPU copy to avoid GPU memory growth
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_to_log(f"[Best so far] Epoch {best_epoch} | ValTotalLoss={best_metric:.6f}",
                        working_dir=working_dir, print_msg=True)

        save_to_log(f"---- Epoch {epoch + 1} Summary ----", working_dir=working_dir, print_msg=True)
        save_to_log(
            f"Training:   Rec Loss: {avg_train_loss_rec:.4f}, Utilization Loss: {avg_train_loss_util:.4f}, Self Entropy Loss: {avg_train_loss_self_entropy:.4f}, Ortho Loss: {avg_train_loss_ortho:.4f}, Utilization: {avg_train_util:.2f}",
            working_dir=working_dir, print_msg=True)
        save_to_log(
            f"Validation: Rec Loss: {avg_val_loss_rec:.4f}, Utilization Loss: {avg_val_loss_util:.4f}, Self Entropy Loss: {avg_val_loss_self_entropy:.4f}, Ortho Loss: {avg_val_loss_ortho:.4f}, Utilization: {avg_val_util:.2f}",
            working_dir=working_dir, print_msg=True)
        save_to_log("----------------------------\n", working_dir=working_dir, print_msg=True)

        # --- 6. Checkpoint Saving ---
        # Save model every 10 epochs
        if (epoch + 1) % 100 == 0 and epoch != args.num_epochs - 1:
            # checkpoint_path = os.path.join(working_dir, f'vqvae_epoch_{epoch + 1}.pt')
            # torch.save(model.state_dict(), checkpoint_path)
            # save_to_log(f"Checkpoint saved to {checkpoint_path}", working_dir=working_dir, print_msg=True)
            tokenize_test(data_root=args.data_root, vec_size=263, model=model,
                          output_path=os.path.join(working_dir, 'motion_tokens_temp'))
            count_codes(data_path=os.path.join(working_dir, 'motion_tokens_temp'), nb_code=args.nb_code,
                        out_path=os.path.join(working_dir, f'code_counts_epoch_{epoch + 1}.csv'))

        # --- Update histories & plot after each epoch ---
        hist_epoch.append(epoch + 1)
        hist_train_rec.append(avg_train_loss_rec)
        hist_val_rec.append(avg_val_loss_rec)
        hist_train_ppl.append(avg_train_util)
        hist_val_ppl.append(avg_val_util)

        update_plot(save_dir=working_dir, lr=args.lr, ratio=args.ratio, quantizer=args.quantizer,
                    hist_epoch=hist_epoch, hist_train_rec=hist_train_rec, hist_val_rec=hist_val_rec,
                    hist_train_util=hist_train_ppl, hist_val_util=hist_val_ppl,
                    fig=fig, ax1=ax1, ax2=ax2, interactive_backend=interactive_backend)

    save_to_log("Training complete!", working_dir=working_dir, print_msg=True)
    # --- Save only the best model found during training ---
    assert best_state is not None, "No best model captured; check training loop."
    final_model_path = os.path.join('checkpoints/', f'dvq-gsst.pt')
    torch.save(best_state, final_model_path)
    save_to_log(f"Best model (epoch {best_epoch}, val_total_loss={best_metric:.6f}) saved to {final_model_path}",
                working_dir=working_dir, print_msg=True)
    # Use the best model to tokenize the dataset and count codes
    model.load_state_dict(best_state)
    tokenize_test(data_root=args.data_root, vec_size=263, model=model,
                  output_path=os.path.join(working_dir, 'motion_tokens'))
    count_codes(data_path=os.path.join(working_dir, 'motion_tokens'), nb_code=args.nb_code,
                out_path=os.path.join(working_dir, f'code_counts_final.csv'))

    save_to_log(f"Final model saved to {final_model_path}", working_dir=working_dir, print_msg=True)
    final_model_path = os.path.join(working_dir, 'final_ckpt.pt')
    torch.save(best_state, final_model_path)
    # remove temp tokenized data
    shutil.rmtree(os.path.join(working_dir, 'motion_tokens_temp'), ignore_errors=True)


if __name__ == '__main__':
    main()
