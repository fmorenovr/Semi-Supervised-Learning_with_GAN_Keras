
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation, ReLU
from keras.layers import GlobalAveragePooling2D, BatchNormalization

from utils.networks import custom_activation

#from tensorflow_addons.layers import WeightNormalization

# Defines a unsupersived output and supervised output.  
# Both shared the same weights until the last Dense.
def define_discriminator_(in_shape=(32,32,3), n_classes=10, learning_rate = 0.0002):
  # image input
  in_image = Input(shape=in_shape)
          
  # downsample 32x32x3 -> 32x32x64
  fe = Conv2D(64, (3, 3), strides=(1,1), padding='same')(in_image) # nn.Conv2d(3, 64, 4, 2, 1, bias=False)
  #fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # downsample 32x32x3 -> 16x16x128
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) # nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
  #fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)

  # downsample 16x16x128 -> 8x8x256
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) # nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
  #fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # downsample 8x8x256 -> 4x4x512
  fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe) # nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
  #fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # downsample 4x4x512 -> 1
  #fe = Conv2D(1, (4,4), strides=(1,1), padding='valid')(fe) # nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
  
  # Flatten feature maps
  fe = Flatten()(fe)
  # Global Pooling feature maps
  #fe = GlobalAveragePooling2D()(fe)
  # dropout
  fe = Dropout(0.4)(fe)
  # output layer nodes
  fe = Dense(n_classes)(fe)
  
  # supervised output
  c_out_layer = Activation('softmax')(fe)
  
  # optimizer
  optimizer_grad = Adam(lr=learning_rate, beta_1=0.5)
  
  # define and compile supervised discriminator model
  supervised_model = Model(in_image, c_out_layer)
  supervised_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer_grad, metrics=['accuracy'])
  
  # unsupervised output
  d_out_layer = Lambda(custom_activation)(fe)
  
  # define and compile unsupervised discriminator model
  unsupervised_model = Model(in_image, d_out_layer)
  unsupervised_model.compile(loss='binary_crossentropy', optimizer=optimizer_grad, metrics=['accuracy'])
  
  return unsupervised_model, supervised_model

