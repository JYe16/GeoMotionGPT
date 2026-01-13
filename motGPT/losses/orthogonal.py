"""
Orthogonality regularization loss for motion token embeddings.

This module encourages motion token embeddings in the LLM's embedding space
to be mutually orthogonal, which can improve the representational quality
and separability of different motion tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_motion_orthogonality_loss(
    embedding_weight: torch.Tensor,
    motion_ids_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
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

    # Check for zero or very small embeddings that could cause NaN after normalization
    norms = motion_embs.norm(dim=-1)
    if (norms < 1e-8).any():
        # Some embeddings have near-zero norm, skip orthogonality loss to avoid NaN
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    # Normalize each embedding to unit length.
    motion_embs = F.normalize(motion_embs, p=2, dim=-1)  # (N, D)
    
    # Check for NaN after normalization (safety check)
    if torch.isnan(motion_embs).any():
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    # Gram matrix of cosine similarities: G_ij = cos(e_i, e_j)
    gram = motion_embs @ motion_embs.t()  # (N, N)

    # We want gram ≈ I: diagonal ~ 1, off-diagonal ~ 0
    eye = torch.eye(gram.size(0), device=device, dtype=gram.dtype)
    diff = gram - eye  # diagonal goes to 0, off-diagonal is cos(e_i, e_j)

    # Only off-diagonal really matters, but diagonal is already zero here.
    # Normalize by N*(N-1) to make it scale invariant w.r.t. number of motion tokens.
    n = gram.size(0)
    loss = diff.pow(2).sum() / (n * (n - 1))
    
    # Final NaN check
    if torch.isnan(loss):
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    return loss


class MotionOrthogonalityLoss(nn.Module):
    """
    A PyTorch module wrapper for orthogonality loss computation.
    
    This can be used as a regularization term during training to encourage
    motion token embeddings to be orthogonal to each other.
    
    Supports two architectures:
    1. MoT: Motion has separate embedding layer, indices are 0..511
    2. Standard: Motion tokens appended to text vocab, indices are original_vocab_size+0..511
    """
    
    def __init__(self, motion_codebook_size: int = 512, lambda_ortho: float = 0.1,
                 original_vocab_size: int = None):
        """
        Args:
            motion_codebook_size: Number of motion tokens in the codebook (typically 512).
            lambda_ortho: Weight for the orthogonality loss term.
            original_vocab_size: Original text vocab size. If None, assumes MoT architecture
                                 where motion indices start at 0.
        """
        super().__init__()
        self.motion_codebook_size = motion_codebook_size
        self.lambda_ortho = lambda_ortho
        self.original_vocab_size = original_vocab_size
        self._motion_ids = None
        self._is_mot_architecture = None  # Will be determined at runtime
    
    def get_motion_token_ids(self, tokenizer=None, is_mot_architecture: bool = None) -> torch.Tensor:
        """
        Get the token IDs for all motion codebook tokens.
        
        Args:
            tokenizer: Unused for MoT, used for standard architecture fallback.
            is_mot_architecture: If True, indices are 0..511. If False, uses original_vocab_size offset.
            
        Returns:
            Tensor of motion token IDs.
        """
        if self._motion_ids is None:
            if is_mot_architecture or (is_mot_architecture is None and self.original_vocab_size is None):
                # MoT architecture: indices are 0..motion_codebook_size-1
                self._motion_ids = torch.arange(self.motion_codebook_size, dtype=torch.long)
            else:
                # Standard architecture: indices are original_vocab_size + 0..motion_codebook_size-1
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
        """
        Compute the weighted orthogonality loss.
        
        Args:
            embedding_weight: The embedding weight matrix from the model.
            tokenizer: Tokenizer (used for standard architecture).
            device: The device to compute on.
            is_mot_architecture: Override architecture detection.
            
        Returns:
            Weighted orthogonality loss.
        """
        # Auto-detect architecture based on embedding size
        if is_mot_architecture is None:
            # If embedding vocab is small (< 1000), it's likely MoT motion embedding
            is_mot_architecture = embedding_weight.shape[0] < 1000
        
        # Reset cached IDs if architecture changed
        if self._is_mot_architecture != is_mot_architecture:
            self._motion_ids = None
            self._is_mot_architecture = is_mot_architecture
        
        motion_ids = self.get_motion_token_ids(tokenizer, is_mot_architecture)
        loss = compute_motion_orthogonality_loss(embedding_weight, motion_ids, device)
        return self.lambda_ortho * loss


def get_embedding_weight_from_model(model) -> torch.Tensor:
    """
    Extract the MOTION embedding weight matrix from the model.
    
    Supports two architectures:
    1. MoT Architecture: Motion tokens in separate pre_processors[1]
    2. Standard Architecture: Motion tokens appended to shared text embedding
    
    Args:
        model: The language model, possibly wrapped by PEFT/LoRA
        
    Returns:
        The embedding weight tensor containing motion tokens.
    """
    # Navigate through PEFT wrapper if present
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base_model = model.base_model.model
    else:
        base_model = model
    
    # Check for MoT architecture (has pre_processors with motion embedding)
    if hasattr(base_model, 'pre_processors') and len(base_model.pre_processors) > 1:
        # MoT Architecture: Return motion embedding from pre_processors[1]
        motion_embedding = base_model.pre_processors[1]
        if hasattr(motion_embedding, 'weight'):
            return motion_embedding.weight
    
    # Standard Architecture: Return shared text embedding (contains motion tokens)
    # For GPT2
    if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'wte'):
        return base_model.transformer.wte.weight
    
    # For LLaMA
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
        return base_model.model.embed_tokens.weight
    
    # For T5
    if hasattr(base_model, 'shared'):
        return base_model.shared.weight
    
    # Fallback: get_input_embeddings
    if hasattr(base_model, 'get_input_embeddings'):
        emb = base_model.get_input_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            return emb.weight
    
    raise ValueError("Cannot find embedding weight in model")
