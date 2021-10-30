from keras.datasets.cifar10 import load_data

from utils.networks import normalize

# load the images CIFAR10
def load_real_samples():
  # load dataset
  (trainX, trainy), (testX, testy) = load_data()
  # expand to 3d, e.g. add channels
  #X = np.expand_dims(trainX, axis=-1)
  #X_test = np.expand_dims(testX, axis=-1)
  testy = testy[:, 0]
  trainy = trainy[:,0]
  X_test = normalize(testX)
  X_train = normalize(trainX)
  print(X_train.shape, trainy.shape, X_test.shape, testy.shape)
  return [X_train, trainy], [X_test, testy]
