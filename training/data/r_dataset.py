import torch.utils.data
from scipy import signal
import numpy as np
from training.utils.stft_local import stft


class RadarDataset(torch.utils.data.Dataset):
    def __init__(self, path, config):
        self.config = config
        allData = np.load(path, allow_pickle=True)

        # This should be modified in accordance with the last modifications on ARIM database
        self.sb_raw = allData[()]['sb']
        self.sb0_fft = np.fft.fft(allData[()]['sb0'], config['no_fft_points']) / config['no_points']
        self.labels = allData[()]['amplitudes']

        # normalize labels
        self.sb0_fft = np.abs(self.sb0_fft)
        if self.config['normalize_labels'] is True:
            self.sb0_fft = self.sb0_fft * (2 / 2.5) - 1

    def __getitem__(self, index):
        if self.config['stft'] is True:
            spectrogram = stft(self.sb_raw[index], 2048, signal.get_window('hamming', 102), 1)
        else:
            spectrogram = signal.spectrogram(self.sb_raw[index], nfft=self.config['no_fft_points'], fs=self.config['fs'],
                                             nperseg=self.config['nperseg'], noverlap=self.config['noverlap'],
                                             window=self.config['window_type'], return_onesided=False, mode='complex')[2]

        return [np.expand_dims(np.abs(spectrogram), 0), self.sb0_fft[index], np.abs(self.labels[index])]

    def __len__(self):
        return len(self.sb0_fft)

    def __repr__(self):
        return self.__class__.__name__
