from keras.datasets.mnist import load_data

import numpy as np

from utils.networks import normalize

# load the images CIFAR10
# load the images MNIST
def load_real_samples():
  # load dataset
  (trainX, trainy), (testX, testy) = load_data()
  # expand to 3d, e.g. add channels
  X_train = np.expand_dims(trainX, axis=-1)
  X_test = np.expand_dims(testX, axis=-1)
  X_test = normalize(X_test)
  X_train = normalize(X_train)
  print(X_train.shape, trainy.shape, X_test.shape, testy.shape)
  return [X_train, trainy], [X_test, testy]
