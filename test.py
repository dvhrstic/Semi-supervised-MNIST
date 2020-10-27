from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
matplotlib.use('agg')


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(np.unique(y_train, return_counts=True))

encoder = OneHotEncoder(sparse=True)
Y_train1 = encoder.fit_transform(y_train.reshape(-1, 1))
Y_test1 = encoder.fit_transform(y_test.reshape(-1, 1))

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print(Y_test.shape)
print(Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
          batch_size=128, epochs=2,
          verbose=2,
          validation_data=(X_test, Y_test))

if not os.path.exists('results'):
    os.makedirs('results')

# saving the model
save_dir = "results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
print(history.history)

mnist_model = load_model('results/keras_mnist.h5')
predicted_classes = mnist_model.predict_classes(X_test)