# This code is based on https://github.com/qiqiApink/MotionGPT.git
import torch.nn as nn
from .encdec import Encoder, Decoder
from .quantize_pq import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset, ProductQuantizer


class PQVAE_148(nn.Module):
    def __init__(self,
                 quantizer='ema_reset',
                 vec_size=148,
                 nb_code=1024,
                 code_dim=512,
                 pq_groups=4,
                 pq_beta=0.1,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = quantizer
        self.encoder = Encoder(vec_size, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm, nb_code=code_dim)
        self.decoder = Decoder(vec_size, code_dim, down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)

        if self.quant == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim)
        elif self.quant == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif self.quant == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif self.quant == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)
        elif self.quant == "pq":
            self.quantizer = ProductQuantizer(nb_code, code_dim, num_groups=pq_groups, beta=pq_beta)
        else:
            raise ValueError(f"Unsupported quantizer: {self.quant}")

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):

        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        ## quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        if x_d.dim() == 2:
            x_d = x_d.unsqueeze(0)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanPQVAE(nn.Module):
    def __init__(self,
                 quantizer='gsst',
                 nb_code=512,
                 vec_size=148,
                 code_dim=512,
                 pq_groups=4,
                 pq_beta=0.1,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        self.nb_joints = 24
        self.pqvae = PQVAE_148(quantizer, vec_size, nb_code, code_dim, pq_groups, pq_beta, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.pqvae.encode(x)  # (N, T)
        return [quants]

    def forward(self, x):
        x_out, loss, perplexity = self.pqvae(x)

        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.pqvae.forward_decoder(x)
        return x_out
