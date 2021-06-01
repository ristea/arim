import argparse

import numpy as np
import scipy.io
import random
import os

random.seed(707)
np.random.seed(707)

NO_TEST_EXAMPLES = 8000


def split_radar_dataset(dataset_path, save_path):
    dataset = scipy.io.loadmat(dataset_path)

    indexes = np.arange(0, len(dataset['sb0_mat']), 1)
    np.random.shuffle(indexes)

    sb0_mat_test = dataset['sb0_mat'][indexes[:NO_TEST_EXAMPLES]]
    sb0_mat_train = dataset['sb0_mat'][indexes[NO_TEST_EXAMPLES:]]

    sb_mat_test = dataset['sb_mat'][indexes[:NO_TEST_EXAMPLES]]
    sb_mat_train = dataset['sb_mat'][indexes[NO_TEST_EXAMPLES:]]

    label_mat_test = dataset['label_mat'][indexes[:NO_TEST_EXAMPLES]]
    label_mat_train = dataset['label_mat'][indexes[NO_TEST_EXAMPLES:]]

    dataset_train = {'sb0': sb0_mat_train,
                     'sb': sb_mat_train,
                     'labels': label_mat_train}

    dataset_test = {'sb0': sb0_mat_test,
                    'sb': sb_mat_test,
                    'labels': label_mat_test}

    np.save(os.path.join(save_path, 'train_radar.npy'), dataset_train)
    np.save(os.path.join(save_path, 'test_radar.npy'), dataset_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARIM DataSet')
    parser.add_argument('--matlab_data_path', '-m', metavar='[path]', type=str,
                        default='./radar_ml_dataset.mat',
                        help="Path to the matlab data set.")
    parser.add_argument('--output_dataset_path', '-o', type=str,
                        default='./',
                        help='The output directory where the processed data will be saved')

    args, _ = parser.parse_known_args()

    split_radar_dataset(args.matlab_data_path, args.output_dataset_path)

