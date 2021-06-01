import torch
import numpy as np
from training.utils.stats_manager import StatsManager
from training.utils.data_logs import save_logs_train, save_logs_eval
import os


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer, config):
        self.config = config
        self.network = network
        self.stats_manager = StatsManager(config)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.experiment_name = config['exp_name']

    def train_epoch(self):
        running_loss = []
        self.network.train()
        for idx, (spectrograms, labels, _) in enumerate(self.train_dataloader, 0):
            if self.config['use_cuda']:
                spectrograms = spectrograms.cuda().float()
                labels = labels.cuda().float()
            else:
                spectrograms = spectrograms.float()
                labels = labels.float()

            self.optimizer.zero_grad()
            predictions = self.network(spectrograms)

            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                print(f'Training loss on iteration {idx} = {running_loss}')
                save_logs_train(os.path.join(self.config['exp_path'], self.experiment_name),
                                f'Training loss on iteration {idx} = {running_loss}')
                running_loss = []

    def eval_net(self, epoch):
        stats_labels = []
        stats_predictions = []

        running_eval_loss = 0.0
        self.network.eval()
        for i, (spectrograms_, labels_, stats_labels_) in enumerate(self.eval_dataloader, 0):
            if self.config['use_cuda']:
                spectrograms_ = spectrograms_.cuda().float()
                labels_ = labels_.cuda().float()
            else:
                spectrograms_ = spectrograms_.float()
                labels_ = labels_.float()

            predictions_ = self.network(spectrograms_)
            eval_loss = self.criterion(predictions_, labels_)
            running_eval_loss += eval_loss.item()

            stats_labels.append(stats_labels_.detach().cpu().numpy())
            stats_predictions.append(predictions_.detach().cpu().numpy())

        print(f'### Evaluation loss on epoch {epoch} = {running_eval_loss}')
        save_logs_eval(os.path.join(self.config['exp_path'], self.experiment_name),
                       f'### Evaluation loss on epoch {epoch} = {running_eval_loss}')

    def train(self):
        try:
            os.mkdir(os.path.join(self.config['exp_path'], self.experiment_name))
        except FileExistsError:
            print("Director already exists! It will be overwritten!")

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch()

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

    def save_net_state(self, epoch):
        path_to_save = os.path.join(self.config['exp_path'], self.experiment_name, 'model_epoch_' + str(epoch) + '.pkl')
        torch.save(self.network, path_to_save)
