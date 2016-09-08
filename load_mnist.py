from numpy.random import permutation
from pandas import read_csv

def load_data(name, nrows=None):
    raw = read_csv('data/' + name, header=None, nrows=nrows)
    y = raw[0]
    raw.drop(0, axis=1, inplace=True)
    print 'Loaded', raw.shape[0], 'patterns with', raw.shape[1], 'features.'
    return raw.values, y.values

def load_train(nrows=None):
    return load_data('mnist_train.csv', nrows=nrows)

def load_test(nrows=None):
    return load_data('mnist_test.csv', nrows=nrows)

def split_val_test(X_and_y, validation_percentage=0.2):
    X, y = X_and_y
    n = X.shape[0]
    shuffled_idx = permutation(n)
    val_size = int(validation_percentage * n)
    return (X[shuffled_idx[:val_size], :], y[shuffled_idx[:val_size]],
            X[shuffled_idx[val_size:], :], y[shuffled_idx[val_size:]])
