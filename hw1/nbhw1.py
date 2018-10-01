import pandas as pd
from sklearn.naive_bayes import MultinomialNB

def format_data(filename, reshape=False):
    tmp = pd.read_csv(filename, header=None)
    tmp1 = tmp.values
    if reshape:
        return tmp1.reshape(tmp.shape[0])
    else:
        return tmp1

X_train = format_data('spam-training.txt')
Y_train = format_data('train-labels.txt', reshape=True)
X_test = format_data('spam-testing.txt')
Y_test = format_data('test-labels.txt', reshape=True)
# print("%d, %d, %d, %d" % (len(X_train), len(Y_train), len(X_test), len(Y_test)))

model = MultinomialNB()
Y_pred = model.fit(X_train, Y_train).predict(X_test)

total = float(len(Y_test))
correct = float((Y_test == Y_pred).sum())
error_pct = 1.0 - (correct/total)
print("Total: %d, Correctly Predicted: %d, Error Percentage: %.4f%%" % (total, correct, error_pct))

