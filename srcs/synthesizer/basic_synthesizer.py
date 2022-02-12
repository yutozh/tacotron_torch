import os
import collections
import numpy as np
from tqdm import tqdm
from srcs.utils import audio
from srcs.model.model import TacotronModel
import torch
from omegaconf import OmegaConf
from srcs.utils import instantiate


class BaseSynthesizer():
    """
    Basic Synthesizer for all models.
    Noted that you need to modify make_test_feature to match your feature type.
    """

    def __init__(self, hparams, args):
        super(BaseSynthesizer, self).__init__()
        self.args = args
        self.hparams = hparams
        self.test_sample = self.get_test_sample()
        # self.test_feature = self.make_test_feature()
        self.load_model()

    def get_test_sample(self):
        test_sample = collections.namedtuple(
            "Test_sample", self.tuple_value)
        return test_sample

    def make_test_feature(self):
        raise NotImplementedError(
            "You must implement your own make_test_feature function.")

    def load_model(self):
        checkpoint = torch.load(self.args.checkpoint)
        loaded_config = OmegaConf.create(checkpoint['config'])
        
        # restore network architecture
        print(loaded_config.arch)
        model = instantiate(loaded_config.arch, hparams=self.hparams)
        print(model)

        # load trained weights
        state_dict = checkpoint['state_dict']
        if loaded_config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # instantiate loss and metrics
        # criterion = instantiate(loaded_config.loss, is_func=True)
        # metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.model.eval()

    def make_feed_dict(self, label_filename=None):
        raise NotImplementedError(
            "You must implement your own make_feed_dict function.")

    def __call__(self, label_filename):
        hparams = self.hparams
        file_id = os.path.splitext(os.path.basename(label_filename))[0]

        with torch.no_grad():
            phones, input_length, acoustic_targets = self.make_feed_dict(label_filename)

            generated_acoustic, _, _ = self.model(phones, input_length, acoustic_targets)

            
            generated_acoustic = generated_acoustic.reshape(-1, hparams.acoustic_dim)

            acoustic_output_path = os.path.join(
                self.args.output_dir, '{}.npy'.format(file_id))
            np.save(acoustic_output_path, generated_acoustic, allow_pickle=False)


        if self.args.use_gl:
            wav = audio.inv_mel_spectrogram(generated_acoustic.T, hparams)

            wav_output_path = os.path.join(
                self.args.output_dir,
                "{}.wav".format(os.path.splitext(os.path.basename(acoustic_output_path))[0]))
            audio.save_wav(wav, wav_output_path, hparams, norm=True)

        return generated_acoustic, acoustic_output_path 

