import torch

def round_up(max_len, r):
    remain = max_len % r
    if remain == 0:
        return max_len
    else:
        return max_len + r - remain 

def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask

def compute_mask(lengths, r, expand_dim=True):
    max_len = torch.max(lengths)
    max_len = round_up(max_len, r)
    if expand_dim:
        return torch.unsqueeze(sequence_mask(lengths, maxlen=max_len,
        dtype=torch.float32), dim=-1)
    return sequence_mask(lengths, maxlen=max_len, dtype=torch.float32)

def MaskedMSE(targets, outputs, targets_lengths, hparams, mask=None):
    '''Computes a masked Mean Squared Error
    '''
    if mask is None:
        mask = compute_mask(targets_lengths, hparams.outputs_per_step, True)

    #[batch_size, time_dimension, channel_dimension(mels)]
    ones = torch.ones(size=[mask.shape[0], mask.shape[1], targets.shape[-1]], dtype=torch.float32)
    mask_ = mask * ones
    # print(outputs.shape)
    # print(targets.shape)

    return (mask_ * (outputs - targets) ** 2).mean()
    # loss = torch.nn.MSELoss()
    # return loss(outputs, targets
    #      weights=mask_)


def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams,
 mask=None):
    '''Computes a masked SigmoidCrossEntropy with logits
    '''
    if mask is None:
        mask = compute_mask(targets_lengths, hparams.outputs_per_step, False)


    losses = targets * -outputs.sigmoid().log() * hparams.cross_entropy_pos_weight + (1 - targets) * -(1 - outputs.sigmoid()).log()



    masked_loss = losses * mask

    return torch.sum(masked_loss) / torch.count_nonzero(mask)