from numpy import pad
import torch
from torch import conv1d
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channel, batch_size, K=16, depth=256):
        super().__init__()

        self.K = K
        self.depth = depth
        self.batch_size = batch_size

        self.conv_channel = 128
        self.projections = [self.conv_channel, input_channel]

        convs = []
        for k in range(1, self.K + 1):
            convs.append(nn.Sequential(nn.Conv1d(input_channel, self.conv_channel, kernel_size=k, padding='same'), nn.ReLU(), nn.BatchNorm1d(self.conv_channel)))
        self.convs_list=nn.ModuleList(convs)

        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=1, stride=1)


        self.proj1 = nn.Sequential(nn.Conv1d(self.conv_channel * self.K , self.projections[0], kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm1d(self.projections[0]))
        self.proj2 = nn.Sequential(nn.Conv1d(self.projections[0], self.projections[1], kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm1d(self.projections[1]))
        
        half_depth = self.depth // 2
        self.before_highway = nn.Linear(input_channel, half_depth)

        # 4-layer HighwayNet:
        self.highwaynet = nn.Sequential(*[Highway(half_depth, half_depth) for _ in range(4)])

        # 双向GRU
        self.gru = nn.GRU(input_size=half_depth, hidden_size=half_depth, bidirectional=True)

    def forward(self, inputs, input_lengths):
        '''
        :param inputs:  (batch_size, sequence_size, embedding_size)
        :param input_lengths:
        :return:
        '''

        # K个1D Conv, 然后将结果拼接
        inputs = inputs.permute(0, 2, 1)
        # print(inputs.shape)
        conv_outputs = torch.concat([conv(inputs) for conv in self.convs_list], dim=1)
        # print(conv_outputs.shape)

        # max pooling
        maxpool_output = self.max_pool(conv_outputs)
        maxpool_output = maxpool_output[:,:,:-1] # maxpooling "same"
        # print(maxpool_output.shape) # torch.Size([45, 2048, 84])

        # Two projection layers:
        proj1_output = F.relu(self.proj1(maxpool_output))
        proj2_output = self.proj2(proj1_output)
        # print(proj2_output.shape) # torch.Size([45, 512, 84])

        # Residual connection:
        highway_input = proj2_output + inputs
        highway_input = highway_input.permute(0, 2, 1) # torch.Size([45, 84, 512])

        half_depth = self.depth // 2
        assert half_depth*2 == self.depth, 'encoder and postnet depths must be even.'

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != half_depth:
            highway_input = self.before_highway(highway_input)
        
        # Highway
        highway_input = self.highwaynet(highway_input)
        rnn_input = highway_input
        # print(rnn_input.shape) # torch.Size([45, 84, 128])

        # Bidirectional RNN
        outputs, (h_n, c_n) = self.gru(rnn_input)
        
        return outputs  # Concat forward and backward


class Highway(nn.Module):
    def __init__(self, input_shape, depth) -> None:
        super().__init__()

        self.linear1 = nn.Sequential(nn.Linear(input_shape, depth), nn.ReLU())
        self.linear2 = nn.Linear(input_shape, depth)
        nn.init.constant_(self.linear2.bias, -1.0)


    def forward(self, inputs):
        H = self.linear1(inputs)
        T = torch.sigmoid(self.linear2(inputs))

        return H * T + inputs * (1.0 - T)

        input_feature = x.shape[-1]
        for i, unit in enumerate(self.prenet_units):
            fc = nn.Linear(input_feature, unit)
            x = fc(x)
            x = F.dropout(x, training=self.is_training)
            input_feature = unit

        return x
