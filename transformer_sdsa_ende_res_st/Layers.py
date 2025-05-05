from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward
from spikingjelly.clock_driven import neuron, functional, surrogate
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn_s = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.slf_attn_t = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )
        self.lif_t = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.lif_s = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        #not spike
        res = enc_input
        enc_input = self.lif_t(enc_input)
        ##time
        enc_input = enc_input.permute(2,1,0,3).contiguous()
        res = res.permute(2,1,0,3).contiguous()
        enc_output, enc_slf_attn = self.slf_attn_t(
            res,enc_input, enc_input, enc_input, mask=mask,mode='t'
        )
        ##space
        enc_output = enc_output.permute(2,1,0,3).contiguous()
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        res = enc_output
        enc_output = self.lif_s(enc_output)
        enc_output, enc_slf_attn = self.slf_attn_s(
            res,enc_output, enc_output, enc_output, mask=mask
        )
        #not spike
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        #not spike
        enc_output = self.pos_ffn(enc_output)
        #not spike
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.lifs = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )
        self.lifs.append(MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy"))
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )
            self.lifs.append(MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy"))

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )
        self.lif0=MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, x):
        x = self.lif0(x)
        x = x.contiguous().transpose(2, 3)
        T,B,D,L=x.shape
        for i in range(len(self.convolutions) - 1):
            x = self.lifs[i](self.convolutions[i](x.flatten(0,1)).reshape(T,B,-1,L).contiguous())
        x = self.convolutions[-1](x.flatten(0,1)).reshape(T,B,-1,L).contiguous()

        x = x.contiguous().transpose(2, 3)
        return x
