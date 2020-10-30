
from keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict


class Data():
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False)
        self.X_train, self.y_train, self.X_test, self.y_test = self.__get_data()
        self.Y_train = self.encoder.fit_transform(self.y_train.reshape(-1, 1))
        self.Y_test = self.encoder.fit_transform(self.y_test.reshape(-1, 1))
        self.labeled_indexes = self.__get_labeled_indexes()

    def __get_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        self.__subset_test_dataset(X_test, y_test)
        return X_train, y_train, X_test, y_test

    def __subset_test_dataset(self, X_test, y_test):
        X_test_subset = []
        y_test_subset = []
        test_dict = defaultdict(list)
        for i, label in enumerate(y_test):
            test_dict[label].append(i)
        for index_list in test_dict.values():
            X_test_subset.extend(X_test[index_list[:100]])
            y_test_subset.extend(y_test[index_list[:100]])
        return X_test_subset, y_test_subset

    def get_test_data(self):
        return self.X_test, self.Y_test

    def __get_labeled_indexes(self):
        return [np.where(self.y_train == label)[0][0] for label in np.unique(self.y_train)]

    def get_labeled_training_data(self):
        X = self.X_train[self.labeled_indexes]
        Y = self.Y_train[self.labeled_indexes]
        return X, Y

    def get_unlabeled_training_data(self):
        mask = np.ones(len(self.X_train), dtype=bool)
        mask[self.labeled_indexes] = False
        return self.X_train[mask, ...], self.Y_train
