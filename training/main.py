import json
import torch
import torch.optim as optim

from training.networks.FConvNet import FConvNet
from training.networks.FConvBigNet import FConvBigNet
from training.trainer import Trainer
from training.data.data_manager import DataManager


def main():
    config = json.load(open('./config.json'))
    data_manager = DataManager(config)

    model = FConvNet()
    if config['use_cuda'] is True:
        model = model.cuda()
    model.apply(FConvNet.init_weights)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    train_loader, validation_loader = data_manager.get_train_eval_dataloaders(config['train_data_path'])

    trainer = Trainer(model, train_loader, validation_loader, criterion, optimizer, config)
    trainer.train()


if __name__ == "__main__":
    main()
