from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from srcs.module.modules import PreNet, PostNet
from srcs.module.encoder import Encoder
from srcs.module.decoder import Decoder
from srcs.module.layers import CustomProjection
from srcs.module.attention import AttAdd
from srcs.text.phones_mix import phone_to_id


class MnistModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TacotronModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        encoder_dim = 256
        decoder_rnn_units = 1024
        self.acoustic_dimension = hparams.acoustic_dim

        self.embedding = nn.Embedding(
            len(phone_to_id), hparams.phone_embedding_dim)

        self.encoder_prenet = PreNet(
            hparams.phone_embedding_dim, prenet_units=[512, 512])

        self.encoder = Encoder(
            input_channel=512, batch_size=hparams.batch_size, K=16, depth=encoder_dim)

        attention_mechanism = AttAdd(
            encoder_dim, decoder_rnn_units, hparams.attention_dim)

        self.decoder = Decoder(decoder_rnn_units=decoder_rnn_units,
                               attention_mechanism=attention_mechanism,
                               input_dim=encoder_dim,
                               output_dim=80,
                               stop_threshold=hparams.stop_threshold,
                               min_iters=10)

        self.postnet = PostNet(input_units=80, postnet_channels=512)

        self.residual_projection = CustomProjection(
            input_units=512, num_units=hparams.acoustic_dim, apply_activation_fn=False)

    def forward(self, inputs, input_length, targets):
        batch_size = inputs.shape[0]
        # print(inputs.shape)  # torch.Size([n, 84])

        embedded_inputs = self.embedding(inputs)
        # print(embedded_inputs.shape)  # torch.Size([n, 84, 512])

        prenet_outputs = self.encoder_prenet(embedded_inputs)
        # print(prenet_outputs.shape)  # torch.Size([n, 84, 512])

        encoder_outputs = self.encoder(prenet_outputs, input_length)
        # print(encoder_outputs.shape)  # torch.Size([45, 84, 256])

        decoder_output, stop_token_output, att_ws = self.decoder(
            memory=encoder_outputs,
            memory_length=input_length,
            targets=targets
        )
        # print(decoder_output.shape) # torch.Size([45, 445, 80])
        # print(stop_token_output.shape) # torch.Size([45, 445])
        # print(att_ws.shape)

        # self.alignments = F.transpose(
        #     final_decoder_state.alignment_history.stack(), [1, 2, 0])

        self.decoder_output = decoder_output.reshape((batch_size, self.acoustic_dimension, -1)) # torch.Size([45, 80, 445])

        self.stop_token_output = stop_token_output.reshape((batch_size, -1))

        # self.stop_token_targets = stop_token_targets
        # print(self.stop_token_targets.shape)

        self.stop_token_binary = torch.sigmoid(stop_token_output)

        # Postnet
        self.postnet_output = self.postnet(self.decoder_output)
        # print(self.postnet_output.shape)

        self.postnet_output = self.postnet_output.permute(0, 2, 1) # torch.Size([45, 445, 512])

        self.projected_output = self.residual_projection(self.postnet_output)

        self.acoustic_outputs = decoder_output + self.projected_output # torch.Size([45, 80, 445])

        self.outputs = self.acoustic_outputs

        return self.outputs, decoder_output, self.stop_token_output
