import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

from ..text.phones_mix import phone_to_id
import hydra

_pad = 0
_stop_token_pad = 1.
_target_pad = -4.1
max_abs_value = 4.

class BiaoBeiDataset(Dataset):
    def __init__(self, data_dir):

        self.label_dir = os.path.join(data_dir, 'labels')
        self.wav_dir = os.path.join(data_dir, 'wavs')
        self.acoustic_dir = os.path.join(data_dir, 'acoustic_features')
        self.wavs = list(os.listdir(self.wav_dir))

    
    def __len__(self):
        return len(self.wavs)
    
    def __getitem__(self, index):
        wav_file_name = self.wavs[index]
        file_id = wav_file_name.split('.')[0]
        
        label_path = os.path.join(self.label_dir, file_id + '.lab')
        acoustic_features_path = os.path.join(self.acoustic_dir, 'mels', file_id + '.npy')

        phones = []
        with open(label_path, 'r', encoding = 'utf-8') as f:
            line = f.readline()
            content = line.strip().split('|')[2].split(' ')
            for item in content:
                phones.append(phone_to_id[item])
        phones.append(phone_to_id["~"])

        acoustic_targets = np.load(acoustic_features_path)

        phones = np.asarray(phones, np.int32)
        input_length = len(phones)
        targets_length = len(acoustic_targets)

        stop_token_targets = [0.0] * (targets_length - 1) + [1.0]
        stop_token_targets = np.asarray(stop_token_targets, np.float32)

        phones = np.squeeze(phones)

        return phones, input_length, acoustic_targets, stop_token_targets, targets_length

def get_collate_fn(outputs_per_step):
    def collate_fn(data_tuple):
        phones = pad_sequence([torch.from_numpy(x[0]) for x in data_tuple], True, _pad)
        input_length = torch.from_numpy(np.asarray([x[1] for x in data_tuple], dtype=np.int32))

        acoustic_targets = torch.from_numpy(_prepare_targets([x[2] for x in data_tuple], outputs_per_step))
        stop_token_targets = torch.from_numpy(_prepare_stop_token_targets([x[3] for x in data_tuple], outputs_per_step))
        targets_length = torch.tensor([x[4] for x in data_tuple], dtype=torch.int32)
        return phones,input_length, acoustic_targets, stop_token_targets, targets_length

    return collate_fn

# def _prepare_inputs(inputs):
#     max_len = max((len(x) for x in inputs))
#     return np.stack([_pad_input(x, max_len) for x in inputs])

# 对齐acoustic feature, 且为outputs_per_step的倍数
def _prepare_targets(targets, alignment):
    max_len = max((len(t) for t in targets))
    return np.stack(
        [_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _prepare_stop_token_targets(targets, alignment):
    max_len = max([len(t) for t in targets])
    return np.stack([
        _pad_stop_token_target(t, _round_up(max_len, alignment))
        for t in targets
    ])


# def _pad_input(x, length):
#     return np.pad(
#         x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    return np.pad(
        t, [(0, length - t.shape[0]), (0, 0)],
        mode='constant',
        constant_values=_target_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


def _pad_stop_token_target(t, length):
    return np.pad(
        t, (0, length - t.shape[0]),
        mode='constant',
        constant_values=_stop_token_pad)