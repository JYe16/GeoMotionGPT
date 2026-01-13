import torch
import torch.nn.functional as F

def generate_motion_orthogonality_loss(embedding_weight: torch.Tensor,
                              motion_ids_tensor: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
    """
    Encourage motion token embeddings to be mutually orthogonal.

    Args:
        embedding_weight: (vocab_size, hidden_dim), model's embedding matrix.
        motion_ids_tensor: (num_motion_tokens,), indices of motion tokens in the vocab.
        device: torch device.

    Returns:
        A scalar tensor representing the orthogonality regularization loss.
    """
    if motion_ids_tensor is None:
        # No motion tokens provided → no regularization.
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    motion_ids = motion_ids_tensor.to(device)
    motion_embs = embedding_weight[motion_ids]  # (N, D)

    # If less than 2 motion tokens, orthogonality is not meaningful.
    if motion_embs.size(0) < 2:
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    # Normalize each embedding to unit length.
    motion_embs = F.normalize(motion_embs, p=2, dim=-1)  # (N, D)

    # Gram matrix of cosine similarities: G_ij = cos(e_i, e_j)
    gram = motion_embs @ motion_embs.t()  # (N, N)

    # We want gram ≈ I: diagonal ~ 1, off-diagonal ~ 0
    eye = torch.eye(gram.size(0), device=device, dtype=gram.dtype)
    diff = gram - eye  # diagonal goes to 0, off-diagonal is cos(e_i, e_j)

    # Only off-diagonal really matters, but diagonal is already zero here.
    # Normalize by N*(N-1) to make it scale invariant w.r.t. number of motion tokens.
    n = gram.size(0)
    loss = diff.pow(2).sum() / (n * (n - 1))

    return loss
