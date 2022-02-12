from turtle import forward
import torch
from torch import conv1d
import torch.nn as nn
import torch.nn.functional as F
from srcs.module.modules import PreNet, OutputProjection, DecoderRNN, StopProjection
from srcs.module.layers import ZoneOutCell

import six


class Decoder(nn.Module):
    def __init__(self,
                 prenet_units=[256, 256],
                 decoder_rnn_layers=2,
                 decoder_rnn_units=1024,
                 prenet_auxiliary_feature=None,
                 rnn_auxiliary_feature=None,
                 dropout_rate=0.2,
                 zoneout_rate=0.1,
                 decoder_rnn_init_state=None,
                 input_dim=0,
                 output_dim=80,
                 outputs_per_step=2,
                 is_training=False,
                 attention_mechanism=None,
                 max_iters=2000,
                 reduction_factor=1,
                 use_concate=True,
                 cumulate_att_w=True,
                 stop_threshold=0.5,
                 min_iters=10):
        super().__init__()

        self.prenet_units = prenet_units
        self.prenet_auxiliary_feature = prenet_auxiliary_feature
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim  # encoder的hidden state的维度
        self.output_dim = output_dim
        self.outputs_per_step = outputs_per_step
        self.is_training = is_training
        self.attention_mechanism = attention_mechanism
        self.max_iters = max_iters
        self.decoder_rnn_layers = decoder_rnn_layers
        self.decoder_rnn_units = decoder_rnn_units
        self.rnn_auxiliary_feature = rnn_auxiliary_feature
        self.decoder_rnn_init_state = decoder_rnn_init_state
        self.zoneout_rate = zoneout_rate
        self.reduction_factor = reduction_factor
        self.use_concate = use_concate
        self.cumulate_att_w=cumulate_att_w
        self._threshold = stop_threshold
        self.min_iters = min_iters

        # define lstm network
        prenet_layers = len(self.prenet_units)
        prenet_units_dim = self.prenet_units[-1] if prenet_layers != 0 else self.output_dim
        self.lstm = torch.nn.ModuleList()
        for layer in six.moves.range(self.decoder_rnn_layers):
            iunits = self.input_dim + prenet_units_dim if layer == 0 else self.decoder_rnn_units
            lstm = torch.nn.LSTMCell(iunits, self.decoder_rnn_units)
            if zoneout_rate > 0.0:
                lstm = ZoneOutCell(lstm, zoneout_rate)
            self.lstm += [lstm]
        
        # define projection layers
        iunits = input_dim + decoder_rnn_units if use_concate else decoder_rnn_units
        self.feat_out = torch.nn.Linear(iunits, output_dim * reduction_factor, bias=False)
        self.prob_out = torch.nn.Linear(iunits, reduction_factor)
        self.stop_out = torch.nn.Linear(iunits, 1)

        self.initialize()

    def forward(self, memory, memory_length, targets=None,
                # targets_length=None, stop_token_targets=None, global_step=None
                ):
        
        batch_size = memory.shape[0]
        # self.helper = self.get_helper(
        #     batch_size,
        #     hparams,
        #     targets=targets,
        #     stop_token_targets=stop_token_targets,
        #     global_step=global_step)

        if self.reduction_factor > 1:
            targets = targets[:, self.reduction_factor -
                              1:: self.reduction_factor]

        hlens = list(map(int, memory_length))
        # print(memory.shape)
        # print(memory_length.shape)
        # print(targets.shape)

        # initialize hidden states of decoder
        c_list = [self._zero_state(memory)]
        z_list = [self._zero_state(memory)]
        for _ in six.moves.range(1, len(self.lstm)):
            c_list += [self._zero_state(memory)]
            z_list += [self._zero_state(memory)]

        # print(c_list[0].shape) # torch.Size([45, 1024])

        # 初始输入 [batch, output_dim]
        prev_out = memory.new_zeros(memory.size(0), self.output_dim)

        # initialize attention
        prev_att_w = None
        self.attention_mechanism.reset()

        # loop for an output sequence
        outs, logits, att_ws, stops = [], [], [], []

        cnt = 0
        while True:
            att_c, att_w = self.attention_mechanism(
                memory, hlens, z_list[0], prev_att_w)  # torch.Size([45, 256]), torch.Size([45, 84])
            prenet_out = self.prenet(
                prev_out) if self.prenet is not None else prev_out
            print(att_c.shape, prenet_out.shape)
            xs = torch.cat([att_c, prenet_out], dim=1) # torch.Size([45, 256+256])

            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))

            for i in six.moves.range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], att_c], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            out = self.feat_out(zcs).view(memory.size(0), self.output_dim, -1)
            logit = self.prob_out(zcs)
            stop = self.stop_out(zcs)

            if not self.training:
                logit = torch.sigmoid(logit)
                stop = torch.sigmoid(stop)

            if self.training == False:
                print(stop.shape, self._threshold)
                if (stop[0][0] > self._threshold and cnt > self.min_iters) or cnt > self.max_iters:
                    break
                prenet_out = out
            else:
                if cnt >= targets.shape[1]:
                    break
                prev_out = targets[:, cnt, :]  # teacher forcing

            outs += [out]
            logits += [logit]
            att_ws += [att_w]
            stops += [stop]

            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            
            cnt += 1
        

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.cat(outs, dim=2)  # (B, odim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)
        stops = torch.cat(stops, dim=1)  # (B, Lmax)

        # print(logits.shape)
        # print(before_outs.shape)
        # print(att_ws.shape)

        before_outs = before_outs.transpose(2, 1)
        return before_outs, stops, att_ws

        # if self.reduction_factor > 1:
        #     before_outs = before_outs.view(
        #         before_outs.size(0), self.odim, -1
        #     )  # (B, odim, Lmax)

        # if self.postnet is not None:
        #     after_outs = before_outs + \
        #         self.postnet(before_outs)  # (B, odim, Lmax)
        # else:
        #     after_outs = before_outs
        # before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        # after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)
        # logits = logits

        # # apply activation function for scaling
        # if self.output_activation_fn is not None:
        #     before_outs = self.output_activation_fn(before_outs)
        #     after_outs = self.output_activation_fn(after_outs)

        # return after_outs, before_outs, logits, att_ws

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def initialize(self):
        self.init_modules()
        self.add_custom_module()

    def init_modules(self):
        """get initial modules"""
        self.prenet = PreNet(
            input_units=self.output_dim,
            prenet_units=self.prenet_units,
            dropout_rate=self.dropout_rate,
            activation_fn=F.relu,
            #  auxiliary_feature=self.prenet_auxiliary_feature,
        )

        # output projection for getting final feature dimensions
        self.output_projection = OutputProjection(
            units=self.output_dim * self.outputs_per_step)

    def add_custom_module(self):
        # Decoder RNN
        self.decoder_rnn = DecoderRNN(num_layers=self.decoder_rnn_layers,
                                      num_units=self.decoder_rnn_units,
                                      zoneout_rate=self.zoneout_rate)

        # Stop projection to predict stop token
        self.stop_token_projection = StopProjection(
            shape=self.outputs_per_step)

    # def get_helper(self, batch_size, hparams, input_length=None,
    #                targets=None, stop_token_targets=None, global_step=None):
    #     if self.is_training:
    #         helper = TrainingHelper(batch_size,
    #                                 targets,
    #                                 stop_token_targets,
    #                                 hparams,
    #                                 global_step)
    #     else:
    #         helper = TestHelper(batch_size, hparams)
    #     return helper
