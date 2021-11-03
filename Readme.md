# SSL-GAN_Keras

Implementarion of Semi-Supervised GANs from the paper ["Improved Techniques for Training GANs"](https://arxiv.org/abs/1606.03498).

To train, just need to install Tensorflow 2 (Im using 2.2 version, stable version in conda). You can try to install using conda:

      conda create --name tf2 python=3.8
      conda activate tf2
      conda install tensorflow-gpu=2.2.0
      conda install cudatoolkit=10.1
      conda install notebook
      conda install keras=2.2.0
      conda serach keras-applications
      conda install keras-applications

To add the Functions over Layers: `WeightNormalization` used in CIFAR10 discriminator/generator network, we use tensorflow-addons 0.11.2, due to the compatibility with Tensorflow 2.2.0, we run:

      pip install -q -U tensorflow-addons==0.11.2

But this command is already writed over the [CIFAR10 notebook](/SSGAN_Keras_CIFAR10.ipynb) in the cell number 3.

### MNIST Results

MNIST is a 10-class dataset containing images in gray scale. The data is composed by images of handwritten number (as you see below).

We train the model using 30 epochs, initial learning rate of 2e-5, Adam optimizer, 128 Batch size, and a label rate of 0.00166 (about 10 labeled samples per class).

You can find the Notebook associate [here](/SSGAN_Keras_MNIST.ipynb). 


| Original  |  Generated  | Loss |
| ------------------- | ------------------- | ------------------- |
|  ![MNIST original](/Results/MNIST/original.png) |  ![MNIST Generated](/Results/MNIST/generated.png) | ![MNIST Loss](/Results/MNIST/GAN_loss.png) |

We achieve 97.190\% in testing:

| Loss  |  Accuracy  |
| ------------------- | ------------------- |
|  ![MNIST Loss](/Results/MNIST/train_test_loss.png) |  ![MNIST Acc](/Results/MNIST/train_test_acc.png) |


### CIFAR10 Results

CIFAR10 is a 10-class dataset containing images in RGB format. The data is composed by images of 10 different kind of objects like horses, frogs, dogs, cats, etc (as you see below).

We train the model using 200 epochs, initial learning rate of 2e-5, Adam optimizer, 128 Batch size, and a label rate of 1.00 (all labeled dataset).

You can find the Notebook associate [here](/SSGAN_Keras_CIFAR10.ipynb).


| Original  |  Generated  | Loss |
| ------------------- | ------------------- | ------------------- |
|  ![CIFAR10 original](/Results/CIFAR10/original.png) |  ![CIFAR10 Generated](/Results/CIFAR10/generated.png) | ![CIFAR10 Loss](/Results/CIFAR10/GAN_loss.png) |

We achieve 93\% in testing:

| Loss  |  Accuracy  |
| ------------------- | ------------------- |
|  ![CIFAR10 Loss](/Results/CIFAR10/train_test_loss.png) |  ![CIFAR10 Acc](/Results/CIFAR10/train_test_acc.png) |

