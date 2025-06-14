import numpy as np
from collections import defaultdict

class StratifiedTrainTestSplitter:
    def __init__(self, test_size=0.2, random_state=None, shuffle=True):
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, X, y):
        np.random.seed(self.random_state)
        X = np.array(X)
        y = np.array(y)

        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_indices[label].append(idx)

        train_idx = []
        test_idx = []

        for label, indices in class_indices.items():
            if self.shuffle:
                np.random.shuffle(indices)
            n_test = int(len(indices) * self.test_size)
            test_idx.extend(indices[:n_test])
            train_idx.extend(indices[n_test:])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        return X_train, X_test, y_train, y_test

    def __call__(self, X, y):
        return self.split(X, y)
