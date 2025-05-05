import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.lif_kv = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, q, k, v, mask=None,mode='s'):

        T,B,L,_=q.shape
        # q=q.flatten(0,1)
        # k=k.flatten(0,1)
        # v=v.flatten(0,1)
        attn = k.mul(v)
        # attn = attn / self.temperature

        if mask is not None and mode == 's':
            attn = attn.masked_fill(mask.unsqueeze(-1), 0)
        if mask is not None and mode == 't':
            attn = attn.masked_fill(mask.transpose(0,2).unsqueeze(-1), 0)

        # attn = self.softmax(attn)
        kv = attn.sum(dim=-2, keepdim=True)
        kv = self.lif_kv(kv)
        output = q.mul(kv)
        # output = torch.bmm(attn, v).view(T,B,L,-1)

        return output, attn
