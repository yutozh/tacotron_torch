from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from srcs.module.layers import ZoneOutCell, conv1d_bn_drop


class PreNet(nn.Module):
    def __init__(self,
                 input_units,
                 prenet_units=[512, 512],
                 dropout_rate=0.1,
                 activation_fn=F.relu):
        super(PreNet, self).__init__()

        self.prenet_units = prenet_units
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        fcs = []
        input_ = input_units
        for unit in prenet_units:
            output_ = unit
            fcs.extend([nn.Linear(input_, output_), nn.ReLU(), nn.Dropout()])
            input_ = output_

        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        x = self.fcs(x)
        return x


class PostNet(nn.Module):
    def __init__(self,
                 input_units,
                 postnet_num_layers=5,
                 postnet_kernel_size=[5, ],
                 postnet_channels=512,
                 postnet_dropout_rate=0.0,
                 activation_fn=torch.tanh
                 ):
        super(PostNet, self).__init__()
        self.postnet_num_layers = postnet_num_layers
        self.postnet_kernel_size = postnet_kernel_size
        self.postnet_channels = postnet_channels
        self.postnet_dropout_rate = postnet_dropout_rate
        self.activation_fn = activation_fn

        first_layer = Conv1d_bn_drop(input_units, kernel_size=self.postnet_kernel_size,
                        channels=self.postnet_channels,
                        activation_fn=self.activation_fn,
                        dropout_rate=self.postnet_dropout_rate)
        mid_layers = [Conv1d_bn_drop(self.postnet_channels, kernel_size=self.postnet_kernel_size,
                        channels=self.postnet_channels,
                        activation_fn=self.activation_fn,
                        dropout_rate=self.postnet_dropout_rate) for i in range(self.postnet_num_layers - 2)]
        self.conv1d_multi = nn.Sequential(*[first_layer, *mid_layers])
        self.conv1d_last = Conv1d_bn_drop(self.postnet_channels, kernel_size=self.postnet_kernel_size,
                                          channels=self.postnet_channels,
                                          activation_fn=lambda _: _,
                                          dropout_rate=self.postnet_dropout_rate)

    def forward(self, inputs):
        x = self.conv1d_multi.forward(inputs)
        x = self.conv1d_last.forward(x)
        return x


class Conv1d_bn_drop(nn.Module):
    def __init__(self, input_channle,
                 kernel_size, channels, activation_fn=None,
                 dropout_rate=0.5):
        super(Conv1d_bn_drop, self).__init__()
        self.activation_fn = activation_fn

        self.conv1d = nn.Conv1d(input_channle, channels,
                                kernel_size=kernel_size, padding="same")
        self.bn = torch.nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):

        outputs = self.conv1d(inputs)
        if self.training:
            outputs = self.bn(outputs)
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)

        outputs = self.dropout(outputs)
        return outputs


class OutputProjection(nn.Module):
    """
    Projection layer to r * acoustic dimensions
    """

    def __init__(self, units=80, activation=None):
        super().__init__()

        self.units = units
        self.activation = activation

    def forward(self, inputs):
        linear = nn.Linear(inputs.shape[-1], self.units)
        output = linear(inputs)

        if self.activate != None:
            output = self.activation_fn(output)

        return output


class DecoderRNN(nn.Module):
    def __init__(self, num_layers=2, num_units=1024, zoneout_rate=0.1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.zoneout_rate = zoneout_rate

    def forward(self, inputs, states):
        self.lstm = torch.nn.LSTMCell(inputs.shape[-1], self.num_units)
        self.zonelstm = ZoneOutCell(self.lstmcell, self.zoneout_rate)

        return self.zonelstm(inputs, states)


class StopProjection(nn.Module):
    """
    Projection to a scalar and through a sigmoid activation
    """

    def __init__(self, shape=1, activation=F.sigmoid):
        super(StopProjection, self).__init__()
        self.shape = shape
        self.activation = activation

    def forward(self, inputs):
        linear = nn.Linear(inputs.shape[-1], self.shape)
        output = linear(inputs)

        # During training, don't use activation as it is integrated inside
        # the sigmoid_cross_entropy loss function
        if self.training:
            return output
        else:
            return self.activation(output)
