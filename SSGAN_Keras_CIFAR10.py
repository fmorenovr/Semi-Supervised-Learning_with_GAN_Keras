#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import time


# In[ ]:


import keras


# In[ ]:


keras.__version__


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[ ]:


#!pip install -q -U tensorflow-addons==0.11.2


# ### Utilities

# In[ ]:


from utils import verifyDir
from utils.networks import normalize, unnormalize, plot_data


# ### Dataset

# In[ ]:


from utils.CIFAR10 import load_real_samples


# ### Discriminator & Generator

# In[ ]:


from utils.CIFAR10 import define_discriminator
from utils.CIFAR10 import define_generator


# ### Semi-Supervised GAN

# In[ ]:


from utils.networks import define_gan


# ### Selecting sub-set 

# In[ ]:


from utils.networks import select_supervised_samples, generate_real_samples
from utils.networks import generate_fake_samples, generate_latent_points


# ### Training

# In[ ]:


from utils.networks import train_gan


# ### Loading Dataset

# In[ ]:


# load image data
dataset_train, dataset_test = load_real_samples()


# ### Parameters

# In[ ]:


input_shape = (32, 32, 3)
num_classes = 10

learning_rate = 2e-4
latent_dim = 100

epochs=100
batch_size=128

labeled_rate = 4/50
labeled_samples = int(dataset_train[0].shape[0]*labeled_rate)


# In[ ]:


LOG_PATH = f"Logs/SSGAN_CIFAR10/Classifier_{labeled_samples}/"
verifyDir(LOG_PATH)


# ### Creating Models

# In[ ]:


from utils.networks import f1_score, auc_pr, precision_score, recall_score


# In[ ]:


metrics_list=["accuracy", f1_score, auc_pr]


# In[ ]:


# create the discriminator models
unsupervised_model, supervised_model = define_discriminator(in_shape=input_shape, 
                                                            n_classes=num_classes, 
                                                            learning_rate = learning_rate,
                                                            metrics_list=metrics_list)
# create the generator
generator_model = define_generator(latent_dim=latent_dim)


# In[ ]:


supervised_model.summary()


# In[ ]:


unsupervised_model.summary()


# In[ ]:


generator_model.summary()


# In[ ]:


# create the gan
from keras.optimizers import Adam
opt_gan = Adam(lr=learning_rate, beta_1=0.5)

gan_model = define_gan(generator_model, unsupervised_model, optimizer_grad = opt_gan)


# In[ ]:


gan_model.summary()


# ### Training

# In[ ]:


train_gan(generator_model, unsupervised_model, supervised_model, gan_model, 
      dataset_train, dataset_test, latent_dim=latent_dim, 
      n_epochs=epochs, n_batch=batch_size, n_classes=num_classes, 
      # n_samples=labeled_samples)
      label_rate=labeled_rate, LOG_PATH=LOG_PATH,
      unnormalize_image=True)


# ### Testing

# In[ ]:


dataset_train, dataset_test = load_real_samples()


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


last_step = int(dataset_train[0].shape[0]/batch_size)*epochs
last_step


# In[ ]:


supervised_model = load_model(f'{LOG_PATH}supervised_model_{last_step}.h5')


# In[ ]:


X_train, y_train = dataset_train
_, acc = supervised_model.evaluate(X_train, y_train, verbose=0)
print('Train Classifier Accuracy: %.3f%%\n' % (acc * 100))


# In[ ]:


X_test, y_test = dataset_test
_, acc = supervised_model.evaluate(X_test, y_test, verbose=0)
print('Test Classifier Accuracy: %.3f%%\n' % (acc * 100))


# ### Plotting

# In[ ]:


import pandas as pd


# In[ ]:


results_file = pd.read_csv(f"{LOG_PATH}SSL_GAN.csv", sep=";")


# In[ ]:


log_file = results_file.iloc[:,1:]
log_file


# In[ ]:


fig = log_file[["generator_loss", "unsupervised_real_loss", "unsupervised_fake_loss", "supervised_loss"]].plot(figsize=(16,12)).get_figure()
fig.savefig(f'{LOG_PATH}GAN_loss.png')


# In[ ]:


fig = log_file[["train_loss", "test_loss"]].plot(figsize=(16,12)).get_figure()
fig.savefig(f'{LOG_PATH}train_test_loss.png')


# In[ ]:


fig = log_file[["train_acc", "test_acc"]].plot(figsize=(16,12), ylim=(0,100), yticks=range(0,110,10)).get_figure()
fig.savefig(f'{LOG_PATH}train_test_acc.png')


# In[ ]:


fig = log_file[["unsupervised_real_acc", "unsupervised_fake_acc"]].plot(figsize=(16,12), ylim=(0,100), yticks=range(0,110,10)).get_figure()
fig.savefig(f'{LOG_PATH}unsupervised_real_fake_acc.png')

