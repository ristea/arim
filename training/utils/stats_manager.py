import numpy as np
from sklearn import metrics


class StatsManager:

    def __init__(self, config):
        self.normalize_labels = config['normalize_labels']

    def get_statistics(self, predictions, labels):
        predictions = np.vstack(predictions)
        labels = np.vstack(labels)

        auc_mean = []
        for idx in range(0, len(predictions)):
            labels[idx] = np.where(np.abs(labels[idx]) == 0, 0, 1)
            if np.sum(np.array(labels[idx])) == 0:
                continue
            fpr, tpr, _ = metrics.roc_curve(labels[idx], predictions[idx])
            auc = metrics.auc(fpr, tpr)

            auc_mean.append(auc)

        auc_mean = np.mean(np.array(auc_mean))
        return auc_mean

