import argparse
import numpy as np
import scipy.io
import random
import os

random.seed(707)
np.random.seed(707)

NO_TEST_EXAMPLES = 8000


def split_dataset(dataset_path):
    dataset = scipy.io.loadmat(dataset_path)

    indexes = np.arange(0, len(dataset['sb0_mat']), 1)
    np.random.shuffle(indexes)

    # Preparing information about signals
    informations = dataset['info_mat']
    info_mat = []
    for i in range(0, len(informations)):
        info_mat.append({'nr_interferences': informations[i][0],
                         'snr': informations[i][1:1+int(informations[i][0])],
                         'sir': informations[i][1+int(informations[i][0]):1 + 2*int(informations[i][0])],
                         'interference_slope': informations[i][1 + 2*int(informations[i][0]):]})
    info_mat = np.array(info_mat)

    sb0_mat_test = dataset['sb0_mat'][indexes[:NO_TEST_EXAMPLES]]
    sb0_mat_train = dataset['sb0_mat'][indexes[NO_TEST_EXAMPLES:]]

    sb_mat_test = dataset['sb_mat'][indexes[:NO_TEST_EXAMPLES]]
    sb_mat_train = dataset['sb_mat'][indexes[NO_TEST_EXAMPLES:]]

    amplitude_mat_test = dataset['amplitude_mat'][indexes[:NO_TEST_EXAMPLES]]
    amplitude_mat_train = dataset['amplitude_mat'][indexes[NO_TEST_EXAMPLES:]]

    distance_mat_test = dataset['distance_mat'][indexes[:NO_TEST_EXAMPLES]]
    distance_mat_train = dataset['distance_mat'][indexes[NO_TEST_EXAMPLES:]]

    info_mat_test = info_mat[indexes[:NO_TEST_EXAMPLES]]
    info_mat_train = info_mat[indexes[NO_TEST_EXAMPLES:]]

    dataset_train = {'sb0': sb0_mat_train,
                     'sb': sb_mat_train,
                     'amplitudes': amplitude_mat_train,
                     'distances': distance_mat_train,
                     'info_mat': info_mat_train}

    dataset_test = {'sb0': sb0_mat_test,
                    'sb': sb_mat_test,
                    'amplitudes': amplitude_mat_test,
                    'distances': distance_mat_test,
                    'info_mat': info_mat_test}

    return dataset_train, dataset_test


def build_radar_dataset(arim_path, save_path):
    arim_train, arim_test = split_dataset(arim_path)

    np.save(os.path.join(save_path, 'arim_train.npy'), arim_train)
    np.save(os.path.join(save_path, 'arim_test.npy'), arim_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARIM Data Set')
    parser.add_argument('--arim_data_path', '-m', metavar='[path]', type=str,
                        default='./',
                        help="Path to the directory with matlab data sets.")
    parser.add_argument('--output_data_path', '-o', type=str,
                        default='./',
                        help='The output directory where the processed data will be saved')

    args, _ = parser.parse_known_args()
    build_radar_dataset(args.arim_data_path, args.output_dataset_path)
