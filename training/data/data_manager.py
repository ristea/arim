from training.data.r_dataset import RadarDataset
import numpy as np
import torch


class DataManager:
    def __init__(self, config):
        self.config = config

    def get_dataloader(self, path):
        dataset = RadarDataset(path, self.config)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=self.config['use_cuda'],
            drop_last=True,
        )
        return dataloader

    def get_train_eval_dataloaders(self, path):
        np.random.seed(707)

        dataset = RadarDataset(path, self.config)
        dataset_size = len(dataset)

        ## SPLIT DATASET
        train_split = self.config['train_eval_split_ratio']
        train_size = int(train_split * dataset_size)
        validation_size = dataset_size - train_size

        ########### CURRENTLY DOING THIS, WHICH WORKS ###########

        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        temp = int(train_size + validation_size)
        val_indices = indices[train_size:temp]

        ## DATA LOARDER ##
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=self.config['use_cuda'],
                                                   drop_last=True)

        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=self.config['use_cuda'],
                                                        drop_last=True)
        return train_loader, validation_loader
