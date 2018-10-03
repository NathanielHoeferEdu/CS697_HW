import logging

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

logger = logging.getLogger('hw3')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

DATASET_FILE = 'boston_housing.txt'
TRAIN_COUNT = 20
ALPHAS = [1e-2, 1e-1, 1, 1e1, 1e2]


def format_data(filename):
    data = pd.read_csv(filename, delimiter=' ', header=None)
    dataset = data.values
    rows = len(dataset)
    cols = len(dataset[0])

    x_train, y_train = dataset[:TRAIN_COUNT, :-1], dataset[:TRAIN_COUNT, -1:]
    x_test, y_test = dataset[TRAIN_COUNT:, :-1], dataset[TRAIN_COUNT:, -1:]
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_rows = len(x_train)
    x_cols = len(x_train[0])
    y_shape = y_train.shape

    logger.debug("Total dataset rows: {}, columns: {}".format(rows, cols))
    logger.debug("Train dataset rows: {}, columns: {}, "
                 "y-shape: {}".format(x_rows, x_cols, y_shape))

    return (x_train, y_train, x_test, y_test)


x_train, y_train, x_test, y_test = format_data(DATASET_FILE)

model = linear_model.RidgeCV(alphas=ALPHAS)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Ridge Regression Model")
print("Dataset file: {}".format(DATASET_FILE))
print("Training set count: {}".format(TRAIN_COUNT))
print("Mean squared error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
print("Alpha selected: {}".format(model.alpha_))
print("Coefficients: {}".format(model.coef_))
