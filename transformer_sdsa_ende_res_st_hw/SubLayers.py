import mindspore.nn as nn
import mindspore.nn.functional as F
import numpy as np

from .Modules import ScaledDotProductAttention
from spikingjelly.clock_driven.surrogate import ATan as atan
from spikingjelly.clock_driven import neuron, functional, surrogate
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.lif_1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.lif_q = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.lif_k = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.lif_v = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.batch_norm1 = nn.BatchNorm1d(n_head * d_k)
        self.batch_norm2 = nn.BatchNorm1d(n_head * d_k)
        self.batch_norm3 = nn.BatchNorm1d(n_head * d_v)

    def forward(self,res, q, k, v, mask=None,mode='s'):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        T,sz_b, len_q, _ = q.size()
        T,sz_b, len_k, _ = k.size()
        T,sz_b, len_v, _ = v.size()
        #q,k,v should be spike
        residual = res

        q = self.w_qs(q).flatten(0,1)
        k = self.w_ks(k).flatten(0,1)
        v = self.w_vs(v).flatten(0,1)
        #(96,133,256)
        k=self.batch_norm1(k.transpose(1,2)).transpose(1,2).reshape(T, sz_b, len_k,-1).contiguous()
        q=self.batch_norm2(q.transpose(1,2)).transpose(1,2).reshape(T, sz_b, len_q,-1).contiguous()
        v=self.batch_norm3(v.transpose(1,2)).transpose(1,2).reshape(T, sz_b, len_v,-1).contiguous()
        #(4,24,132,256)
        k = k.view(T,sz_b, len_k, n_head, d_k)
        v = v.view(T,sz_b, len_v, n_head, d_v)
        q = q.view(T,sz_b, len_q, n_head, d_k)
        #(4,12,132,2,128)
        q = q.permute(0,3, 1, 2, 4).contiguous().view(T,-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(0,3, 1, 2, 4).contiguous().view(T,-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(0,3, 1, 2, 4).contiguous().view(T,-1, len_v, d_v)  # (n*b) x lv x dv
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)

        mask = mask.repeat(1,n_head, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask,mode=mode)

        output = output.view(T,n_head, sz_b, len_q, d_v)
        output = (
            output.permute(0,2, 3, 1, 4).contiguous().view(T,sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        # output = self.dropout(self.fc(output))
        output = self.fc(output)
        # output = self.lif_1(output)
        output = self.layer_norm(output+residual)
        # output = output + residual
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        self.lif_0 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.lif_1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.lif_2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, x):
        residual = x
        #residual should be spike
        #(4,24,117,256)
        x = self.lif_0(x)
        output = x.transpose(2, 3)
        T,B,D,L = output.shape 
        output = output.flatten(0,1)
        #(96,256,117)
        output = self.w_2(self.lif_1(self.w_1(output).reshape(T,B,-1,L).contiguous()).flatten(0,1))
        # output = self.lif_2(output)
        output = output.reshape(T,B,-1,L).transpose(2, 3).contiguous()
        # output = self.dropout(output)
        output = self.layer_norm(output + residual)
        # output = output + residual

        return output
