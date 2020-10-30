from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import sys
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model, Sequential
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, y_train, X_test, y_test


# If you need at least 100 test labeled
# images of each class
# you can iterate through np.where()[0][i]
# where 0 < i < 9
def get_labeled_indexes(y_train):
    return [np.where(y_train == x)[0][0] for x in np.unique(y_train)]


def get_test_data():
    return 


def define_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adam')
    return model


def train_model(X_train, Y_train):
    model = KerasClassifier(build_fn=define_model, verbose=1)
    #epochs = [1, 5, 10, 20]
    #batch_sizes = [32, 64, 128, 256]
    epochs = [5]
    batch_sizes = [128]
    param_grid = dict(epochs=epochs, batch_size=batch_sizes)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, Y_train)
    return grid_result.best_estimator_


def predict_model(keras_model_wrapper, X):
    probabilities = keras_model_wrapper.model.predict(X)
    new_labels = encoder.fit_transform(np.argmax(
                                       probabilities, axis=-1).reshape(-1, 1))
    return new_labels


def evaluate_model(keras_model_wrapper, X, Y):
    metrics = keras_model_wrapper.model.evaluate(X, Y, verbose=0)
    return metrics


X_train, y_train, X_test, y_test = get_data()
labeled_indexes = get_labeled_indexes(y_train)

encoder = OneHotEncoder(sparse=False)
Y_train = encoder.fit_transform(y_train.reshape(-1, 1))
Y_test = encoder.fit_transform(y_test.reshape(-1, 1))

X = X_train[labeled_indexes]
Y = Y_train[labeled_indexes]

model = train_model(X, Y)
mask = np.ones(len(X_train), dtype=bool)
mask[labeled_indexes] = False

pseudo_labels = predict_model(model, X_train[mask, ...])

test_scores_labeled_data = evaluate_model(model, X_test, Y_test)

# Training 2
finalized_model = train_model(np.concatenate((X_train[mask, ...], X)),
                              np.concatenate((pseudo_labels, Y)))

final_labels = predict_model(model, X_test)

test_scores_all_data = evaluate_model(finalized_model, X_test, Y_test)
print("Test data accuracy for model trained on labeled data ",
      test_scores_labeled_data[1])
print("Test data accuracy for model trained on all data ",
      test_scores_all_data[1])
