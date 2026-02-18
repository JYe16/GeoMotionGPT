import argparse
import os
import shutil
import sys
from datetime import datetime

project_root = '../'
sys.path.append(project_root)

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

from dvq.utils.define_device import define_device
from dvq.utils.training_related import save_to_log
from dvq.data_preprocessing.tokenize_dataset import tokenize as tokenize_test
from dvq.utils.code_count import count_codes
from dvq.utils.graph import update_plot
from transformers import get_cosine_schedule_with_warmup

from dvq.dataloader.humanml3d.humanml3d_263_dataset_mgpt import HumanML3DDataset
from dvq.dataloader.kit.kit_dataset import KITDataset
from dvq.utils.define_num_workers import define_num_workers
from pqvae.pq_vae import HumanPQVAE

# Use non-GUI backend if DISPLAY is not available (e.g., on servers)
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_CONFIGS = {
    "humanml3d": {"data_root": "datasets/humanml3d/", "vec_size": 263},
    "kit": {"data_root": "/data/jackieye/KIT-ML/", "vec_size": 251},
}


def main():
    parser = argparse.ArgumentParser(description="PQ-VAE codebook training script")
    parser.add_argument('--dataset', type=str, default='humanml3d', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--codebook_lr', type=float, default=2e-5)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--nb_code', type=int, default=512)
    parser.add_argument('--code_dim', type=int, default=512)
    parser.add_argument('--pq_groups', type=int, default=4)
    parser.add_argument('--pq_beta', type=float, default=0.1)
    parser.add_argument('--commit_weight', type=float, default=0.25)
    parser.add_argument('--commit_clip', type=float, default=10.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=define_num_workers())
    parser.add_argument('--quantizer', type=str, default='pq', choices=['pq'])

    args = parser.parse_args()

    if args.code_dim % args.pq_groups != 0:
        raise ValueError(f"code_dim ({args.code_dim}) must be divisible by pq_groups ({args.pq_groups})")

    ds_cfg = DATASET_CONFIGS[args.dataset]
    if args.data_root is None:
        args.data_root = ds_cfg["data_root"]
    vec_size = ds_cfg["vec_size"]

    now = datetime.now()
    exp_tag = (
        f"{now.strftime('%m%d%Y')}-{args.dataset}-pq"
        f"-g{args.pq_groups}-c{args.nb_code}-d{args.code_dim}"
    )
    working_dir = os.path.join('experiments', 'pqvae', exp_tag)
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    device = define_device()
    save_to_log(f"Using device: {device}", working_dir=working_dir, print_msg=True)
    save_to_log(
        f"Dataset: {args.dataset} (vec_size={vec_size}, data_root={args.data_root})",
        working_dir=working_dir,
        print_msg=True,
    )
    save_to_log(
        f"PQ config: nb_code={args.nb_code}, code_dim={args.code_dim}, pq_groups={args.pq_groups}, pq_beta={args.pq_beta}",
        working_dir=working_dir,
        print_msg=True,
    )

    if args.dataset == "kit":
        DatasetClass = KITDataset
    else:
        DatasetClass = HumanML3DDataset

    train_set = DatasetClass(data_root=args.data_root, split="train", window_size=args.window_size)
    effective_bs = min(args.batch_size, len(train_set))
    train_loader = DataLoader(
        train_set,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_set = DatasetClass(data_root=args.data_root, split="val", window_size=args.window_size)
    val_bs = min(args.batch_size, len(val_set))
    val_loader = DataLoader(
        val_set,
        batch_size=val_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = HumanPQVAE(
        nb_code=args.nb_code,
        vec_size=vec_size,
        quantizer=args.quantizer,
        code_dim=args.code_dim,
        pq_groups=args.pq_groups,
        pq_beta=args.pq_beta,
        down_t=3,
    ).to(device)

    with_decay, no_decay, codebook_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "quantizer.codebooks" in n:
            codebook_params.append(p)
        elif n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            with_decay.append(p)

    optim_groups = [
        {"params": with_decay, "lr": args.lr, "weight_decay": 1e-4},
        {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
    ]
    if len(codebook_params) > 0:
        optim_groups.append({"params": codebook_params, "lr": args.codebook_lr, "weight_decay": 0.0})

    optimizer = optim.AdamW(optim_groups)

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    save_to_log("Starting PQ-VAE training loop...", working_dir=working_dir, print_msg=True)

    plt.ion()
    interactive_backend = (matplotlib.get_backend().lower() != "agg")
    hist_epoch, hist_train_rec, hist_val_rec = [], [], []
    hist_train_ppl, hist_val_ppl = [], []
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    best_metric = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(args.num_epochs):
        model.train()
        train_rec, train_commit, train_ppl = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            x_hat, loss_commit, perplexity = model(batch)
            loss_rec = F.mse_loss(x_hat, batch)
            loss_commit_used = torch.clamp(loss_commit, max=args.commit_clip)
            loss = loss_rec + args.commit_weight * loss_commit_used

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            train_rec += loss_rec.item()
            train_commit += loss_commit.item()
            train_ppl += perplexity.item()

            pbar.set_postfix({
                'rec': f'{loss_rec.item():.4f}',
                'commit': f'{loss_commit.item():.4f}',
                'commit_used': f'{loss_commit_used.item():.4f}',
                'ppl': f'{perplexity.item():.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })

        avg_train_rec = train_rec / len(train_loader)
        avg_train_commit = train_commit / len(train_loader)
        avg_train_ppl = train_ppl / len(train_loader)
        avg_train_commit_weighted = args.commit_weight * min(avg_train_commit, args.commit_clip)

        model.eval()
        val_rec, val_commit, val_ppl = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, loss_commit, perplexity = model(batch)
                loss_rec = F.mse_loss(x_hat, batch)

                val_rec += loss_rec.item()
                val_commit += loss_commit.item()
                val_ppl += perplexity.item()

        avg_val_rec = val_rec / len(val_loader)
        avg_val_commit = val_commit / len(val_loader)
        avg_val_ppl = val_ppl / len(val_loader)
        avg_val_commit_weighted = args.commit_weight * min(avg_val_commit, args.commit_clip)

        val_total = avg_val_rec + args.commit_weight * avg_val_commit
        if val_total < best_metric:
            best_metric = val_total
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_to_log(
                f"[Best so far] Epoch {best_epoch} | ValTotalLoss={best_metric:.6f}",
                working_dir=working_dir,
                print_msg=True,
            )

        save_to_log(f"---- Epoch {epoch + 1} Summary ----", working_dir=working_dir, print_msg=True)
        save_to_log(
            f"Training:   Rec={avg_train_rec:.4f}, Commit={avg_train_commit:.4f}, CommitWeighted={avg_train_commit_weighted:.4f}, Perplexity={avg_train_ppl:.2f}",
            working_dir=working_dir,
            print_msg=True,
        )
        save_to_log(
            f"Validation: Rec={avg_val_rec:.4f}, Commit={avg_val_commit:.4f}, CommitWeighted={avg_val_commit_weighted:.4f}, Perplexity={avg_val_ppl:.2f}",
            working_dir=working_dir,
            print_msg=True,
        )
        save_to_log("----------------------------\n", working_dir=working_dir, print_msg=True)

        if (epoch + 1) % 250 == 0 and epoch != args.num_epochs - 1:
            tokenize_test(
                data_root=args.data_root,
                vec_size=vec_size,
                model=model.pqvae,
                output_path=os.path.join(working_dir, 'motion_tokens_temp'),
            )
            count_codes(
                data_path=os.path.join(working_dir, 'motion_tokens_temp'),
                nb_code=args.nb_code,
                out_path=os.path.join(working_dir, f'code_counts_epoch_{epoch + 1}.csv'),
            )

        hist_epoch.append(epoch + 1)
        hist_train_rec.append(avg_train_rec)
        hist_val_rec.append(avg_val_rec)
        hist_train_ppl.append(avg_train_ppl)
        hist_val_ppl.append(avg_val_ppl)

        update_plot(
            save_dir=working_dir,
            lr=args.lr,
            ratio=float(args.pq_groups),
            quantizer=args.quantizer,
            hist_epoch=hist_epoch,
            hist_train_rec=hist_train_rec,
            hist_val_rec=hist_val_rec,
            hist_train_util=hist_train_ppl,
            hist_val_util=hist_val_ppl,
            fig=fig,
            ax1=ax1,
            ax2=ax2,
            interactive_backend=interactive_backend,
        )

    save_to_log("Training complete!", working_dir=working_dir, print_msg=True)
    assert best_state is not None, "No best model captured; check training loop."

    ckpt_stem = f"{args.dataset}-pqvae-{args.quantizer}-g{args.pq_groups}-c{args.nb_code}-d{args.code_dim}"
    final_model_path = os.path.join('checkpoints', f'{ckpt_stem}.pt')
    torch.save(best_state, final_model_path)
    save_to_log(
        f"Best model (epoch {best_epoch}, val_total_loss={best_metric:.6f}) saved to {final_model_path}",
        working_dir=working_dir,
        print_msg=True,
    )

    model.load_state_dict(best_state)
    tokenize_test(
        data_root=args.data_root,
        vec_size=vec_size,
        model=model.pqvae,
        output_path=os.path.join(working_dir, 'motion_tokens'),
    )
    count_codes(
        data_path=os.path.join(working_dir, 'motion_tokens'),
        nb_code=args.nb_code,
        out_path=os.path.join(working_dir, 'code_counts_final.csv'),
    )

    run_ckpt_path = os.path.join(working_dir, f'final_ckpt_{ckpt_stem}.pt')
    torch.save(best_state, run_ckpt_path)
    save_to_log(f"Run checkpoint saved to {run_ckpt_path}", working_dir=working_dir, print_msg=True)

    shutil.rmtree(os.path.join(working_dir, 'motion_tokens_temp'), ignore_errors=True)


if __name__ == '__main__':
    main()
