# This code is based on https://github.com/qiqiApink/MotionGPT.git
import torch.nn as nn
from .encdec import Encoder, Decoder
from .quantize_softgumbel import SoftGumbelQuantizer


class VQVAE_148(nn.Module):
    def __init__(self,
                 quantizer='gsst',
                 vec_size=148,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 ratio=6.0,
                 self_entropy_ratio=0.2,
                 norm=None):

        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = quantizer
        self.encoder = Encoder(vec_size, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm, nb_code=nb_code)
        self.decoder = Decoder(vec_size, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)

        self.quantizer = SoftGumbelQuantizer(
            nb_code=nb_code,
            code_dim=code_dim,
            ratio=ratio,
            self_entropy_ratio=self_entropy_ratio
        )

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
        x_quantized, loss_util, loss_self_entropy, perplexity = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss_util, loss_self_entropy, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAE(nn.Module):
    def __init__(self,
                 quantizer='gsst',
                 nb_code=512,
                 vec_size=148,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 ratio=6.0,
                 norm=None,
                 self_entropy_ratio=0.2):
        super().__init__()

        self.nb_joints = 24
        self.vqvae = VQVAE_148(quantizer, vec_size, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, ratio=ratio, norm=norm, self_entropy_ratio=self_entropy_ratio)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x)  # (N, T)
        return [quants]

    def forward(self, x):
        x_out, loss_util, loss_self_entropy, perplexity = self.vqvae(x)

        return x_out, loss_util, loss_self_entropy, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
