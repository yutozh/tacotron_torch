from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from .dataset import BiaoBeiDataset, get_collate_fn


def get_data_loaders(data_config, training=True):

    dataset = BiaoBeiDataset(data_config.data_dir)

    loader_args = {
        'batch_size': data_config.batch_size,
        'shuffle': data_config.shuffle,
        'num_workers': data_config.num_workers
    }

    collate_fn = get_collate_fn(data_config.outputs_per_step)

    if training:
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(data_config.validation_split, int):
            assert data_config.validation_split > 0
            assert data_config.validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = data_config.validation_split
        else:
            num_valid = int(num_total * data_config.validation_split)
        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])
        return DataLoader(train_dataset, collate_fn=collate_fn, **loader_args), \
               DataLoader(valid_dataset, **loader_args)
    else:
        return DataLoader(dataset, **loader_args)

