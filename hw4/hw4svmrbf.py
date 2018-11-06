import logging

import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

logger = logging.getLogger('hw4')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

TRAIN_DATASET = 'zip.train'
TEST_DATASET = 'zip.test'
KERNEL = 'rbf'
CV_FOLDS = 5
C_VALUES = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]


def format_data(filename):
    data = pd.read_csv(filename, delim_whitespace=True,
                       skipinitialspace=True, header=None)
    dataset = data.values
    labels, entries = dataset[:, :1], dataset[:, 1:]
    labels = labels.flatten().astype(int)

    x_rows = len(entries)
    x_cols = len(entries[0])
    x_shape = entries.shape
    y_len = len(labels)
    y_shape = labels.shape

    logger.debug("{} entry rows: {}, columns: {}, "
                 "shape: {}".format(filename, x_rows, x_cols, x_shape))
    logger.debug(" -- First 10 features of first entry: {}".format(entries[0][:10]))
    logger.debug("{} labels count: {}, shape: {}".format(filename, y_len, y_shape))
    logger.debug(" -- First 10 labels: {}".format(labels[:10]))

    return (entries, labels)


if __name__ == '__main__':
    x_train, y_train = format_data(TRAIN_DATASET)
    x_test, y_test = format_data(TEST_DATASET)

    print("Calculating best C value using {} kernel via 5-fold "
          "cross validation accuracy:".format(KERNEL))
    model = svm.SVC(kernel=KERNEL, gamma='auto')
    max_c = 0
    max_cv_score = 0
    for c in C_VALUES:
        model.C = c
        cv_scores = cross_val_score(model, x_train, y_train, cv=CV_FOLDS)
        mean_cv_score = np.mean(cv_scores)
        print("C value: {:<6} - CV Accuracies: {}, Average Accuracy: "
              "{}".format(c, cv_scores, mean_cv_score))
        if mean_cv_score > max_cv_score:
            max_cv_score = mean_cv_score
            max_c = c
    print("C value selected: {}\n".format(max_c))

    model.C = max_c
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    total = len(y_test)
    correct = sum(a == b for a,b in zip(y_test, y_pred))
    error_pct = (1.0 - (float(correct) / float(total))) * 100

    print("Support Vector Machine using {} kernel and C value "
          "of {}".format(KERNEL, max_c))
    print(" -- Total: %d, Correctly Predicted: %d" % (total, correct))
    print(" -- Error Percentage: %.4f%%" % (error_pct))


