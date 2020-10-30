from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
import tensorflow as tf
import logging
import numpy as np
from sklearn.preprocessing import OneHotEncoder
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class NeuralNetworkModel():
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False)

    def define_model(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(784,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                           metrics=['accuracy'], optimizer='adam')
        return model

    def train(self, X_train, Y_train):
        wrapper_model = KerasClassifier(build_fn=self.define_model, verbose=1)
        epochs = [5, 15]
        batch_sizes = [128, 256]
        param_grid = dict(epochs=epochs, batch_size=batch_sizes)
        grid = GridSearchCV(estimator=wrapper_model, param_grid=param_grid)
        grid_result = grid.fit(X_train, Y_train)
        self.model = grid_result.best_estimator_.model

    def predict(self, X):
        probabilities = self.model.predict(X)
        new_labels = self.encoder.fit_transform(np.argmax(
                                        probabilities, axis=-1).reshape(-1, 1))
        return new_labels

    def evaluate(self, X, Y):
        metrics = self.model.evaluate(X, Y, verbose=0)
        return metrics
