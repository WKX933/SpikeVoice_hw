import mindspore
import mindspore.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
# from transformer.Layers import FFTBlock as fft_dec
from text.symbols import symbols
from spikingjelly.clock_driven.surrogate import ATan as atan
from spikingjelly.clock_driven import neuron, functional, surrogate
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return mindspore.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        self.T = config["T"]
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.atan = atan()
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.time_embed = nn.Parameter(mindspore.zeros(1, self.T, d_word_vec))

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        src_seq = (src_seq.unsqueeze(0)).repeat(self.T, 1, 1)
        #(4,12,124)
        T,batch_size, max_len = src_seq.shape

        # -- Prepare masks
        mask = mask.unsqueeze(0).repeat(T,1,1)
        slf_attn_mask = mask.unsqueeze(2).expand(-1,-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            print('length of src larger than max_length!!')
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[2], self.d_model
            )[: src_seq.shape[2], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq)             
            enc_output = enc_output+self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        ###time embedding
        enc_output=enc_output.permute(2,1,0,3).contiguous()
        enc_output = enc_output + self.time_embed.repeat(batch_size, 1, 1)
        enc_output=enc_output.permute(2,1,0,3).contiguous()

        # enc_output = self.atan(enc_output)
        for enc_layer in self.layer_stack:
            #not spike
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output




class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]
        self.T = config["T"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.time_embed = nn.Parameter(mindspore.zeros(1, self.T, d_word_vec))

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        #enc_seq(24,864,256)
        T,batch_size, max_len,D = enc_seq.shape
        # batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            print('seq_len out of range!!!!!!!!!!!!!')
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[2], self.d_model
            )[: enc_seq.shape[2], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            #mask(24,866)
            mask = mask.unsqueeze(0).repeat(T,1,1)
            slf_attn_mask = mask.unsqueeze(2).expand(-1,-1, max_len, -1)
            # slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            #enc_seq(24,866,256)
            dec_output = enc_seq[:,:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

            dec_output=dec_output.permute(2,1,0,3).contiguous()
            dec_output = dec_output + self.time_embed.repeat(batch_size, 1, 1)
            dec_output=dec_output.permute(2,1,0,3).contiguous()
            mask = mask[:,:, :max_len]
            slf_attn_mask = slf_attn_mask[:,:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask
