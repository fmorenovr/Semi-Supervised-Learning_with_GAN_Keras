from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation, ReLU
from keras.layers import GlobalAveragePooling2D, BatchNormalization

from utils.networks import custom_activation

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(28,28,1), n_classes=10, learning_rate = 0.0001, metrics_list=["accuracy"]):
  # image input
  in_image = Input(shape=in_shape)
  # downsample
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
  fe = LeakyReLU(alpha=0.2)(fe)
  # downsample
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  # downsample
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  # flatten feature maps
  fe = Flatten()(fe)
  # dropout
  fe = Dropout(0.4)(fe)
  # output layer nodes
  fe = Dense(n_classes)(fe)
  
  # supervised output
  c_out_layer = Activation('softmax')(fe)
  
  # optimizer
  optimizer_grad = Adam(lr=learning_rate, beta_1=0.15)

  # define and compile supervised discriminator model
  supervised_model = Model(in_image, c_out_layer)
  supervised_model.compile(loss='categorical_crossentropy', optimizer=optimizer_grad, metrics=metrics_list)
  
  # unsupervised output
  d_out_layer = Lambda(custom_activation)(fe)
  
  # define and compile unsupervised discriminator model
  unsupervised_model = Model(in_image, d_out_layer)
  unsupervised_model.compile(loss='binary_crossentropy', optimizer=optimizer_grad, metrics=['accuracy'])
  
  return unsupervised_model, supervised_model
