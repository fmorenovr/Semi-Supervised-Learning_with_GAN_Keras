
import numpy as np
from numpy.random import randn, randint

# select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_classes=10, n_samples=None, label_rate=None):
  X, y = dataset
  X_list, y_list = list(), list()
  if n_samples is not None:
    n_per_class = int(n_samples / n_classes)
  for i in range(n_classes):
    # get all images for this class
    X_with_class = X[y == i]
    # choose random instances
    if n_samples is not None:
      ix = randint(0, len(X_with_class), n_per_class)
    if label_rate is not None:
      ix = randint(0, len(X_with_class), int(len(X_with_class)*label_rate))
    # add to list
    [X_list.append(X_with_class[j]) for j in ix]
    [y_list.append(i) for j in ix]
  return np.asarray(X_list), np.asarray(y_list)


# select real samples
def generate_real_samples(dataset, n_samples=100):
  # split into images and labels
  images, labels = dataset
  # choose random instances
  ix = randint(0, images.shape[0], n_samples)
  # select images and labels
  X, labels = images[ix], labels[ix]
  # generate class labels
  y = np.ones((n_samples, 1))
  return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples=100):
  # generate points in the latent space
  z_input = randn(latent_dim * n_samples)
  # reshape into a batch of inputs for the network
  z_input = z_input.reshape(n_samples, latent_dim)
  return z_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples=100):
  # generate points in latent space
  z_input = generate_latent_points(latent_dim, n_samples)
  # predict outputs
  images = generator.predict(z_input)
  # create class labels
  y = np.zeros((n_samples, 1))
  return images, y



