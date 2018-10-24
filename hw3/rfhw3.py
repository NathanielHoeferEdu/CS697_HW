import logging

import pandas as pd
from sklearn import ensemble

logger = logging.getLogger('hw3')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

TRAIN_DATASET = 'zip.train'
TEST_DATASET = 'zip.test'
N_ESTIMATORS = [1, 3, 5, 10, 20, 100]


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

def train_random_forest(n_estimators, x_train, y_train,
                        x_test, y_test):
    model = ensemble.RandomForestClassifier(n_estimators=n_estimators)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    total = len(y_test)
    correct = sum(a == b for a,b in zip(y_test, y_pred))
    error_pct = (1.0 - (float(correct) / float(total))) * 100

    print("Random Forest Classifier Model with {} "
          "estimators".format(n_estimators))
    print(" -- Total: %d, Correctly Predicted: %d" % (total, correct))
    print(" -- Error Percentage: %.4f%%" % (error_pct))


if __name__ == '__main__':
    x_train, y_train = format_data(TRAIN_DATASET)
    x_test, y_test = format_data(TEST_DATASET)

    for num in N_ESTIMATORS:
        train_random_forest(num, x_train, y_train, x_test, y_test)