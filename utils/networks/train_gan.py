import numpy as np

from .functions import plot_data, normalize, unnormalize
from .dataset import *

# train the generator and discriminator
def train_gan(generator_model, unsupervised_model, supervised_model, gan_model, 
              dataset_train, dataset_test, 
              latent_dim=100, n_epochs=20, n_batch=100, 
              n_classes=10, label_rate=None, n_samples=None, 
              unnormalize_image=False, LOG_PATH=""):
    
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
    
    f_history.write("step;generator_loss;"\
                    "unsupervised_real_loss;unsupervised_real_acc;"\
                    "unsupervised_fake_loss;unsupervised_fake_acc;"\
                    "supervised_loss;supervised_acc;"\
                    "train_loss;test_loss;"\
                    "train_mse;test_mse;"\
                    "train_f1;test_f1;"\
                    "train_acc;test_acc\n")
    
    #for epoch in n_epochs:
    #    for batch in range(bat_per_epo):
    for step in range(1,n_steps+1):
#         t_start = time.time()
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], n_samples=n_batch)
        c_loss, c_acc, _, _ = supervised_model.train_on_batch(Xsup_real, ysup_real)
        
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
        loss_train, acc_train, mse_train, f1_train = supervised_model.evaluate(X_train, y_train, verbose=0)

        # evaluate the test classifier model
        X_test, y_test = dataset_test
        loss_test, acc_test, mse_test, f1_test = supervised_model.evaluate(X_test, y_test, verbose=0)
        
        # Log
        print('epoch: %d | step: %d | Train: G_Loss: %.3f, ' \
              'D_unsup_loss_real: %.3f, D_unsup_acc_real:  %.2f, ' \
              'D_unsup_loss_fake: %.3f, D_unsup_acc_fake: %.2f, ' \
              'D_sup_loss: %.3f, D_sup_acc: %.2f ' \
              'Train f1: %.3f Test f1: %.3f ' \
              'Train acc: %.3f Test acc: %.3f ' %(int(step/bat_per_epo), step, g_loss,
                                                d_loss1, real_acc*100,
                                                d_loss2, fake_acc*100,
                                                c_loss, c_acc*100,
                                                f1_train, f1_test,
                                                acc_train*100, acc_test*100))#, end = '\r')
        
        f_history.write(f"{step};{g_loss};"\
                        f"{d_loss1};{real_acc*100};"\
                        f"{d_loss2};{fake_acc*100};"\
                        f"{c_loss};{c_acc*100};"\
                        f"{loss_train};{loss_test};"\
                        f"{mse_train};{mse_test};"\
                        f"{f1_train};{f1_test};"\
                        f"{acc_train*100};{acc_test*100}\n")
        
        if step==1:
            # prepare fake examples
            X_generated, _ = generate_fake_samples(generator_model, latent_dim, n_samples=100)

            if unnormalize_image:
              # plot real samples
              plot_data(unnormalize(X_test).astype(int), 0, "test", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
              # scale from [-1,1] to [0,1]
              plot_data(unnormalize(X_generated).astype(int), step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
            else:
              # plot real samples
              plot_data(X_test, 0, "test", grid_size = [10, 10], OUT_PATH=LOG_PATH, gray=True)
            
              # scale from [-1,1] to [0,1]
              plot_data(X_generated, step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH, gray=True)
            
            # evaluate train set
            X_train, y_train = dataset_train
            _, acc, _, f1 = supervised_model.evaluate(X_train, y_train, verbose=1)
            print('Train Classifier Accuracy: %.3f%%, F1: %.3f%% \n' % (acc * 100, f1))
            
            # evaluate the test classifier model
            X_test, y_test = dataset_test
            _, acc, _, f1 = supervised_model.evaluate(X_test, y_test, verbose=1)
            print('Test Classifier Accuracy: %.3f%%, F1: %.3f%% \n' % (acc * 100, f1))
            
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

            if unnormalize_image:
              # scale from [-1,1] to [0,1]
              plot_data(unnormalize(X_generated).astype(int), step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
            else:
              # scale from [-1,1] to [0,1]
              plot_data(X_generated, step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH, gray=True)
            
        elif (step) % (bat_per_epo) == 0:
            # prepare fake examples
            X_generated, _ = generate_fake_samples(generator_model, latent_dim, n_samples=100)

            if unnormalize_image:
              # scale from [-1,1] to [0,1]
              plot_data(unnormalize(X_generated).astype(int), step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH)
            
            else:
              # scale from [-1,1] to [0,1]
              plot_data(X_generated, step, "generated", grid_size = [10, 10], OUT_PATH=LOG_PATH, gray=True)
            
            # evaluate train set
            X_train, y_train = dataset_train
            _, acc, _, f1 = supervised_model.evaluate(X_train, y_train, verbose=1)
            print('Train Classifier Accuracy: %.3f%%, F1: %.3f%% \n' % (acc * 100, f1))
            
            # evaluate the test classifier model
            X_test, y_test = dataset_test
            _, acc, _, f1 = supervised_model.evaluate(X_test, y_test, verbose=1)
            print('Test Classifier Accuracy: %.3f%%, F1: %.3f%% \n' % (acc * 100, f1))
            
            # save the generator model
            filename2 = f'{LOG_PATH}generator_model_{step}.h5'
            generator_model.save(filename2)
            # save the classifier model
            filename3 = f'{LOG_PATH}supervised_model_{step}.h5'
            supervised_model.save(filename3)
            
            print('>Saving models Generator: %s and Supervised: %s' % (filename2, filename3))
    
    f_history.close()
