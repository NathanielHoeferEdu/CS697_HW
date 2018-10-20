import logging

import pandas as pd
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

logger = logging.getLogger('hw3')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

TRAIN_DATASET = 'zip.train'
TEST_DATASET = 'zip.test'
NEAREST_NEIGHBORS = [3, 5, 7]


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

def train_k_nearest_neighbors(nearest_neighbor_num, x_train, y_train,
                              x_test, y_test):
    model = neighbors.KNeighborsClassifier(n_neighbors=nearest_neighbor_num)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("K-Nearest Neighbor Classifier Model with {} nearest "
          "neighbors".format(nearest_neighbor_num))
    print(" -- Mean squared error: "
          "{:.6f}".format(mean_squared_error(y_test, y_pred)))


if __name__ == '__main__':
    x_train, y_train = format_data(TRAIN_DATASET)
    x_test, y_test = format_data(TEST_DATASET)

    for num in NEAREST_NEIGHBORS:
        train_k_nearest_neighbors(num, x_train, y_train, x_test, y_test)