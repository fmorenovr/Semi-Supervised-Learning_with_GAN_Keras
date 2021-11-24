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


# train the generator and discriminator
def train_gan(generator_model, unsupervised_model, supervised_model, gan_model, 
              dataset_train, dataset_test, 
              latent_dim=100, n_epochs=20, n_batch=100, 
              n_classes=10, label_rate=None, n_samples=None):
    
    # select supervised dataset_train
    X_sup, y_sup = select_supervised_samples(dataset_train, 
                                             n_classes=n_classes, 
                                             label_rate=label_rate, 
                                             n_samples=n_samples)
    
    print("Supervised samples:", X_sup.shape, y_sup.shape)
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset_train[0].shape[0] / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print('n_epochs=%d, n_batch=%d, batch/epoch=%d, steps=%d' % (n_epochs, n_batch, bat_per_epo, n_steps))
    
    # manually enumerate epochs
    f_history = open(f"{LOG_PATH}SSL_GAN.csv", "w")
    
    f_history.write("step;generator_loss;"                    "unsupervised_real_loss;unsupervised_real_acc;"                    "unsupervised_fake_loss;unsupervised_fake_acc;"                    "supervised_loss;supervised_acc;"                    "train_loss;test_loss;"                    "train_mse;test_mse;"                    "train_auc;test_auc;"                    "train_f1;test_f1;"                    "train_acc;test_acc\n")
    
    #for epoch in n_epochs:
    #    for batch in range(bat_per_epo):
    for step in range(1,n_steps+1):
#         t_start = time.time()
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], n_samples=n_batch)
        c_loss, c_acc, _, _, _ = supervised_model.train_on_batch(Xsup_real, ysup_real)
        
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset_train, n_samples=n_batch)
        d_loss1, real_acc = unsupervised_model.train_on_batch(X_real, y_real)
        
        X_fake, y_fake = generate_fake_samples(generator_model, latent_dim, n_samples=n_batch)
        d_loss2, fake_acc = unsupervised_model.train_on_batch(X_fake, y_fake)
        
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_samples=n_batch), np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
#         t_total = (time.time() - t_start)
        # summarize loss on this batch
    
        # Train - Test
        X_train, y_train = dataset_train
        loss_train, acc_train, mse_train, f1_train, auc_train = supervised_model.evaluate(X_train, y_train, verbose=0)

        # evaluate the test classifier model
        X_test, y_test = dataset_test
        loss_test, acc_test, mse_test, f1_test, auc_test = supervised_model.evaluate(X_test, y_test, verbose=0)
        
        # Log
        print('epoch: %d | step: %d | Train: G_Loss: %.3f, '               'D_unsup_loss_real: %.3f, D_unsup_acc_real:  %.2f, '               'D_unsup_loss_fake: %.3f, D_unsup_acc_fake: %.2f, '               'D_sup_loss: %.3f, D_sup_acc: %.2f '               'Train auc: %.3f Test auc: %.3f '               'Train f1: %.3f Test f1: %.3f '               'Train acc: %.3f Test acc: %.3f ' %(int(step/bat_per_epo), step, g_loss,
                                                d_loss1, real_acc*100,
                                                d_loss2, fake_acc*100,
                                                c_loss, c_acc*100,
                                                auc_train, auc_test,
                                                f1_train, f1_test,
                                                acc_train*100, acc_test*100))#, end = '\r')
        
        f_history.write(f"{step};{g_loss};"                        f"{d_loss1};{real_acc*100};"                        f"{d_loss2};{fake_acc*100};"                        f"{c_loss};{c_acc*100};"                        f"{loss_train};{loss_test};"                        f"{mse_train};{mse_test};"                        f"{auc_train};{auc_test};"                        f"{f1_train};{f1_test};"                        f"{acc_train*100};{acc_test*100}\n")
        
        if step==1:
            # plot real samples
            plot_data(unnormalize(X_test).astype(int), 0, "test", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
            # prepare fake examples
            X_generated, _ = generate_fake_samples(generator_model, latent_dim, n_samples=100)
            # scale from [-1,1] to [0,1]
            plot_data(unnormalize(X_generated).astype(int), step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
            # evaluate train set
            X_train, y_train = dataset_train
            _, acc, _, f1, auc = supervised_model.evaluate(X_train, y_train, verbose=1)
            print('Train Classifier Accuracy: %.3f%%, F1: %.3f%%, AUC: %.3f%% \n' % (acc * 100, f1, auc))
            
            # evaluate the test classifier model
            X_test, y_test = dataset_test
            _, acc, _, f1, auc = supervised_model.evaluate(X_test, y_test, verbose=1)
            print('Test Classifier Accuracy: %.3f%%, F1: %.3f%%, AUC: %.3f%% \n' % (acc * 100, f1, auc))
            
            # save the generator model
            filename2 = f'{LOG_PATH}generator_model_{step}.h5'
            generator_model.save(filename2)
            # save the classifier model
            filename3 = f'{LOG_PATH}supervised_model_{step}.h5'
            supervised_model.save(filename3)
            
            print('>Saving models Generator: %s and Supervised: %s' % (filename2, filename3))
        
        elif (step) % (100) == 0:
            # prepare fake examples
            X_generated, _ = generate_fake_samples(generator_model, latent_dim, n_samples=100)
            # scale from [-1,1] to [0,1]
            plot_data(unnormalize(X_generated).astype(int), step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
        elif (step) % (bat_per_epo) == 0:
            # prepare fake examples
            X_generated, _ = generate_fake_samples(generator_model, latent_dim, n_samples=100)
            # scale from [-1,1] to [0,1]
            plot_data(unnormalize(X_generated).astype(int), step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
            # evaluate train set
            X_train, y_train = dataset_train
            _, acc, _, f1, auc = supervised_model.evaluate(X_train, y_train, verbose=1)
            print('Train Classifier Accuracy: %.3f%%, F1: %.3f%%, AUC: %.3f%% \n' % (acc * 100, f1, auc))
            
            # evaluate the test classifier model
            X_test, y_test = dataset_test
            _, acc, _, f1, auc = supervised_model.evaluate(X_test, y_test, verbose=1)
            print('Test Classifier Accuracy: %.3f%%, F1: %.3f%%, AUC: %.3f%% \n' % (acc * 100, f1, auc))
            
            # save the generator model
            filename2 = f'{LOG_PATH}generator_model_{step}.h5'
            generator_model.save(filename2)
            # save the classifier model
            filename3 = f'{LOG_PATH}supervised_model_{step}.h5'
            supervised_model.save(filename3)
            
            print('>Saving models Generator: %s and Supervised: %s' % (filename2, filename3))
    
    f_history.close()


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
      label_rate=labeled_rate)


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

