import torch.nn.functional as F
from srcs.module.ops import MaskedMSE, MaskedSigmoidCrossEntropy

def nll_loss(output, target):
    return F.nll_loss(output, target)


def loss(acoustic_outputs, decoder_output, stop_token_output, acoustic_targets, stop_token_targets, targets_length, hparams):
    before_loss = MaskedMSE(acoustic_targets,
                            decoder_output,
                            targets_length,
                            hparams)
    # Compute loss of predictions after postnet
    after_loss = MaskedMSE(acoustic_targets,
                           acoustic_outputs,
                           targets_length,
                           hparams)

    # Compute <stop_token> loss
    stop_token_loss = MaskedSigmoidCrossEntropy(
        stop_token_targets,
        stop_token_output,
        targets_length,
        hparams=hparams)

    loss = before_loss + after_loss + stop_token_loss
    return loss
