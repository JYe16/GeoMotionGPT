import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity



class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        
        N, width, T = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity

    def quantize(self, z):

        assert z.shape[-1] == self.e_dim

        # B x V
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):

        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x



class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.reset_codebook()
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))
        
    def reset_codebook(self):
        self.init = False
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = nn.Parameter(out[:self.nb_code])
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_count = code_count  # nb_code
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()

        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape
        # Preprocess
        x = self.preprocess(x)
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)
        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity

    
class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = code_update
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity


class ProductQuantizer(nn.Module):
    def __init__(self, nb_code, code_dim, num_groups=4, beta=0.25, mu=0.99):
        super().__init__()
        if code_dim % num_groups != 0:
            raise ValueError(f"code_dim ({code_dim}) must be divisible by num_groups ({num_groups}).")

        self.nb_code = nb_code
        self.code_dim = code_dim
        self.num_groups = num_groups
        self.group_dim = code_dim // num_groups
        self.beta = beta
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.register_buffer('codebooks', torch.zeros(self.num_groups, self.nb_code, self.group_dim))
        self.register_buffer('code_sum', torch.zeros(self.num_groups, self.nb_code, self.group_dim))
        self.register_buffer('code_count', torch.zeros(self.num_groups, self.nb_code))

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def _split_groups(self, x):
        return x.view(x.shape[0], self.num_groups, self.group_dim)

    def _tile(self, x_group):
        nb_code_x, dim = x_group.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(dim)
            out = x_group.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x_group
        return out

    @torch.no_grad()
    def init_codebook(self, x_groups):
        for group_idx in range(self.num_groups):
            out = self._tile(x_groups[:, group_idx, :])
            init_cb = out[:self.nb_code]
            self.codebooks[group_idx] = init_cb
            self.code_sum[group_idx] = init_cb
            self.code_count[group_idx] = torch.ones(self.nb_code, device=x_groups.device)
        self.init = True

    @torch.no_grad()
    def update_codebook(self, x_groups, code_idx):
        perplexities = []
        for group_idx in range(self.num_groups):
            x_g = x_groups[:, group_idx, :]
            idx_g = code_idx[:, group_idx]

            one_hot = F.one_hot(idx_g, self.nb_code).float()
            code_count = one_hot.sum(dim=0)
            code_sum = torch.matmul(one_hot.t(), x_g)

            self.code_sum[group_idx] = self.mu * self.code_sum[group_idx] + (1.0 - self.mu) * code_sum
            self.code_count[group_idx] = self.mu * self.code_count[group_idx] + (1.0 - self.mu) * code_count

            denom = self.code_count[group_idx].unsqueeze(-1).clamp_min(1e-5)
            code_update = self.code_sum[group_idx] / denom

            rand_bank = self._tile(x_g)[:self.nb_code]
            usage = (self.code_count[group_idx] >= 1.0).float().unsqueeze(-1)
            self.codebooks[group_idx] = usage * code_update + (1.0 - usage) * rand_bank

            probs = code_count / code_count.sum().clamp_min(1e-8)
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            perplexities.append(perplexity)

        return torch.stack(perplexities).mean()

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        perplexities = []
        for group_idx in range(self.num_groups):
            indices = code_idx[:, group_idx]
            one_hot = F.one_hot(indices, self.nb_code).float()
            probs = one_hot.mean(dim=0)
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            perplexities.append(perplexity)
        return torch.stack(perplexities).mean()

    def quantize(self, x):
        x_groups = self._split_groups(x)
        all_indices = []
        for group_idx in range(self.num_groups):
            x_g = x_groups[:, group_idx, :]
            c_g = self.codebooks[group_idx]
            distance = (
                torch.sum(x_g ** 2, dim=-1, keepdim=True)
                - 2 * torch.matmul(x_g, c_g.t())
                + torch.sum(c_g ** 2, dim=-1).unsqueeze(0)
            )
            _, code_idx = torch.min(distance, dim=-1)
            all_indices.append(code_idx)
        return torch.stack(all_indices, dim=-1)

    def dequantize(self, code_idx):
        if code_idx.dim() == 1:
            if code_idx.numel() % self.num_groups != 0:
                raise ValueError("Flattened PQ indices length must be divisible by num_groups.")
            code_idx = code_idx.view(-1, self.num_groups)
        elif code_idx.dim() == 2 and code_idx.shape[-1] != self.num_groups:
            if code_idx.shape[-1] % self.num_groups != 0:
                raise ValueError("Last dimension of PQ indices must be divisible by num_groups.")
            code_idx = code_idx.view(-1, self.num_groups)
        elif code_idx.dim() > 2 and code_idx.shape[-1] != self.num_groups:
            raise ValueError("PQ indices last dimension must equal num_groups.")

        original_shape = code_idx.shape[:-1]
        flat_idx = code_idx.reshape(-1, self.num_groups)

        outputs = []
        for group_idx in range(self.num_groups):
            outputs.append(F.embedding(flat_idx[:, group_idx], self.codebooks[group_idx]))
        x = torch.cat(outputs, dim=-1)
        return x.view(*original_shape, self.code_dim)

    def forward(self, x):
        N, _, T = x.shape
        x_flat = self.preprocess(x)
        x_groups = self._split_groups(x_flat)

        if self.training and not self.init:
            self.init_codebook(x_groups)

        code_idx = self.quantize(x_flat)
        x_q = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x_groups, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        loss = torch.mean((x_q - x_flat.detach()) ** 2)
        if self.beta > 0:
            loss = loss + self.beta * torch.mean((x_q.detach() - x_flat) ** 2)

        x_q = x_flat + (x_q - x_flat).detach()
        x_q = x_q.view(N, T, -1).permute(0, 2, 1).contiguous()
        return x_q, loss, perplexity
