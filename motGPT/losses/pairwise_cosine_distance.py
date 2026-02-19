"""
Pairwise cosine-distance regularization loss for motion token embeddings.

This module encourages motion token embeddings in the LLM's embedding space
to have large pairwise cosine distance, which improves separability among
different motion tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_motion_pairwise_cosine_distance_loss(
    embedding_weight: torch.Tensor,
    motion_ids_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Encourage motion token embeddings to maximize pairwise cosine distance.

    Args:
        embedding_weight: (vocab_size, hidden_dim), model's embedding matrix.
        motion_ids_tensor: (num_motion_tokens,), indices of motion tokens in the vocab.
        device: torch device.

    Returns:
        A scalar tensor representing the pairwise cosine-distance regularization loss.
    """
    if motion_ids_tensor is None:
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    motion_ids = motion_ids_tensor.to(device)
    motion_embs = embedding_weight[motion_ids]  # (N, D)

    if motion_embs.size(0) < 2:
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    norms = motion_embs.norm(dim=-1)
    if (norms < 1e-8).any():
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    motion_embs = F.normalize(motion_embs, p=2, dim=-1)  # (N, D)
    if torch.isnan(motion_embs).any():
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    gram = motion_embs @ motion_embs.t()  # (N, N)
    eye_mask = torch.eye(gram.size(0), device=device, dtype=torch.bool)
    off_diag = gram[~eye_mask]

    # Minimize mean off-diagonal cosine similarity <=> maximize pairwise cosine distances.
    loss = off_diag.mean()

    if torch.isnan(loss):
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    return loss


class MotionPairwiseCosineDistanceLoss(nn.Module):
    """
    A PyTorch module wrapper for pairwise cosine-distance loss computation.

    Supports two architectures:
    1. MoT: Motion has separate embedding layer, indices are 0..511
    2. Standard: Motion tokens appended to text vocab, indices are original_vocab_size+0..511
    """

    def __init__(self, motion_codebook_size: int = 512, lambda_pairwise: float = 0.1,
                 original_vocab_size: int = None):
        super().__init__()
        self.motion_codebook_size = motion_codebook_size
        self.lambda_pairwise = lambda_pairwise
        self.original_vocab_size = original_vocab_size
        self._motion_ids = None
        self._is_mot_architecture = None

    def get_motion_token_ids(self, tokenizer=None, is_mot_architecture: bool = None) -> torch.Tensor:
        if self._motion_ids is None:
            if is_mot_architecture or (is_mot_architecture is None and self.original_vocab_size is None):
                self._motion_ids = torch.arange(self.motion_codebook_size, dtype=torch.long)
            else:
                start_idx = self.original_vocab_size if self.original_vocab_size else 0
                self._motion_ids = torch.arange(
                    start_idx,
                    start_idx + self.motion_codebook_size,
                    dtype=torch.long
                )
        return self._motion_ids

    def forward(
        self,
        embedding_weight: torch.Tensor,
        tokenizer,
        device: torch.device,
        is_mot_architecture: bool = None
    ) -> torch.Tensor:
        if is_mot_architecture is None:
            is_mot_architecture = embedding_weight.shape[0] < 1000

        if self._is_mot_architecture != is_mot_architecture:
            self._motion_ids = None
            self._is_mot_architecture = is_mot_architecture

        motion_ids = self.get_motion_token_ids(tokenizer, is_mot_architecture)
        loss = compute_motion_pairwise_cosine_distance_loss(embedding_weight, motion_ids, device)
        return self.lambda_pairwise * loss
