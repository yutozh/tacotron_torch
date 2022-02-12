import os
import argparse
import collections
import numpy as np
from tqdm import tqdm
from srcs.utils.utils import *
from srcs.text.feature_converter import *
from srcs.synthesizer.basic_synthesizer import BaseSynthesizer
from srcs.data_loader.data_loaders import get_data_loaders
import torch

class TacotronSynthesizer(BaseSynthesizer):
    """Synthesizer for Tacotron model"""

    def __init__(self, hparams, args):
        self.tuple_value = "phones input_length acoustic_targets targets_length stop_token_targets"
        BaseSynthesizer.__init__(self, hparams, args)

    # def make_test_feature(self, data):
    #     phones, input_length, acoustic_targets, stop_token_targets, targets_length = data
    #     # phones = tf.placeholder(tf.int32, [1, None], 'phones')
    #     # input_length = tf.placeholder(tf.int32, [1], 'input_length')
    #     # acoustic_targets = None #  tf.placeholder(tf.float32, [1, None, self.hparams.acoustic_dim], 'acoustic_targets')

    #     return self.test_sample(
    #         phones=phones,
    #         input_length=input_length,
    #         acoustic_targets=acoustic_targets,
    #         targets_length=None,
    #         stop_token_targets=None)

    def make_feed_dict(self, label_filename):
        hparams = self.hparams
        phones= label_to_sequence(label_filename)
        # setup data_loader instances

        input_length = torch.from_numpy(np.asarray([len(phones)]))
        phones = torch.from_numpy(np.asarray(phones, np.int32)).unsqueeze(0)
        print(phones.shape)
        print("----")
        return phones, input_length, None
