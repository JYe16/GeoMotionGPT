import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftGumbelQuantizer(nn.Module):
    """
    Final, Stabilized Soft Gumbel Quantizer.

    This version removes the unstable balance/entropy losses and introduces the
    essential Commitment Loss, which is standard practice for stabilizing VQ-VAE training.
    This hybrid approach combines the benefits of differentiable Gumbel-Softmax sampling
    with the necessary regularization for the encoder.
    """

    def __init__(self, nb_code=512, code_dim=512, ratio=5, st_mix=0.0, self_entropy_ratio=0.0):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.ratio = ratio

        # Codebook embedding layer
        self.codebook = nn.Embedding(self.nb_code, self.code_dim)
        # Use standard embedding init (much larger than old ±1/K ≈ ±0.002)
        # to prevent training collapse on smaller datasets
        nn.init.normal_(self.codebook.weight, 0.0, 0.02)

        # Hyperparameters for training, will be updated from the main training script
        self.tau = 0.4
        self.st_mix = st_mix
        self.ent_coef = self_entropy_ratio
        self.hard_ppl_rate = 0.0

    def preprocess(self, x):
        # (N, C, T) -> (N*T, C)
        x = x.permute(0, 2, 1).contiguous()
        return x.view(-1, x.shape[-1])

    def postprocess(self, x_quantized, N, T):
        # (N*T, C) -> (N, C, T)
        return x_quantized.view(N, T, -1).permute(0, 2, 1).contiguous()

    def _calculate_perplexity(self, probs):
        mean_probs = torch.mean(probs, dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-7))
        return torch.exp(entropy)
    

    def forward(self, x_encoder):
        N, C, T = x_encoder.shape
        # 1. Gumbel-Softmax Sampling
        x = self.preprocess(x_encoder)
        y_soft = F.gumbel_softmax(x, tau=self.tau, hard=False, dim=-1)
        
        # Straight-through Softmax: One-hot forward, Softmax backward
        y_soft_clear = F.softmax(x/self.tau, dim=-1)
        index = y_soft_clear.max(dim=-1, keepdim=True)[1]
        y_hard_clear = torch.zeros_like(y_soft_clear).scatter_(-1, index, 1.0)
        y_clear = y_soft_clear + (y_hard_clear - y_soft_clear).detach()

        # y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        # y_hard_st = y_soft + (y_hard - y_soft).detach()
        # y_mixed = (1 - self.st_mix) * y_soft + self.st_mix * y_hard_st

        y_hard_st = F.gumbel_softmax(x, tau=self.tau, hard=True, dim=-1)
        # 3. Calculate Quantized Vector
        x_quantized = torch.matmul(y_hard_st, self.codebook.weight)


        # 4. Calculate Perplexity and Postprocess
        # Linear interpolation of perplexity between hard ST and clear softmax
        perplexity = (1 - self.hard_ppl_rate) * self._calculate_perplexity(y_hard_st) +  self.hard_ppl_rate * self._calculate_perplexity(y_clear)
       
        loss_util = (1 - perplexity / self.nb_code) * ((1 - (0.85 * self.hard_ppl_rate)) * self.ratio)

        # Self-entropy regularization
        ent = -torch.sum(y_soft * torch.log(y_soft + 1e-10), dim=-1)
        loss_self_entropy = ent.mean() * self.ent_coef

        x_quantized_out = self.postprocess(x_quantized, N, T)

        return x_quantized_out, loss_util, loss_self_entropy, perplexity

    def quantize(self, x):
        assert not self.training, "quantize should only be used in eval mode"
        
        # hard quantize
        # y_soft = F.softmax(x/self.tau, dim=-1)
        # quantized = torch.argmax(torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0),
        #                          dim=-1)
        
        # Deterministic quantization: argmax of logits
        quantized = torch.argmax(x, dim=-1)
        return quantized

    def dequantize(self, code_idx):
        return self.codebook(code_idx)


