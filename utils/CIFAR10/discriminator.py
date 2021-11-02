
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation, ReLU
from keras.layers import GlobalAveragePooling2D, BatchNormalization

from utils.networks import custom_activation

#from tensorflow_addons.layers import WeightNormalization

# Defines a unsupersived output and supervised output.  
# Both shared the same weights until the last Dense.

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(32,32,3), n_classes=10, learning_rate = 0.0001):
  # image input
  in_image = Input(shape=in_shape)
          
  # downsample 32x32x3 -> 32x32x64
  fe = Conv2D(64, (3, 3), strides=(1,1), padding='same')(in_image) # nn.Conv2d(3, 64, 4, 2, 1, bias=False)
  fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # downsample 32x32x3 -> 16x16x128
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) # nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
  fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)

  # downsample 16x16x128 -> 8x8x256
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) # nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
  fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # downsample 8x8x256 -> 4x4x512
  fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe) # nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
  fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # downsample 4x4x512 -> 1
  #fe = Conv2D(1, (4,4), strides=(1,1), padding='valid')(fe) # nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
  
  # Flatten feature maps
  fe = Flatten()(fe)
  # Global Pooling feature maps
  #fe = GlobalAveragePooling2D()(fe)
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
  unsupervised_model.compile(loss='binary_crossentropy', optimizer=optimizer_grad)
  
  return unsupervised_model, supervised_model



# define the standalone supervised and unsupervised discriminator models
def define_discriminator_paper(in_shape=(32,32,3), n_classes=10):
  # image input
  in_image = Input(shape=in_shape)
  # downsample 32x32 -> 16x16
  fe = Dropout(0.2)(in_image)
  fe = WeightNormalization(Conv2D(96, (3,3), strides=(1,1), padding='same'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = WeightNormalization(Conv2D(96, (3,3), strides=(1,1), padding='same'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = WeightNormalization(Conv2D(96, (3,3), strides=(2,2), padding='same'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = Dropout(0.5)(fe)
  
  # downsample 16x16 -> 8x8
  fe = WeightNormalization(Conv2D(192, (3,3), strides=(1,1), padding='same'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = WeightNormalization(Conv2D(192, (3,3), strides=(1,1), padding='same'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = WeightNormalization(Conv2D(192, (3,3), strides=(2,2), padding='same'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = Dropout(0.5)(fe)
  
  # downsample 8x8 -> 4x4
  fe = WeightNormalization(Conv2D(192, (3,3), strides=(1,1), padding='valid'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = WeightNormalization(Conv2D(192, (1,1), strides=(1,1), padding='valid'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = WeightNormalization(Conv2D(192, (1,1), strides=(1,1), padding='valid'), data_init=False)(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  
  # Global Pooling feature maps
  fe = GlobalAveragePooling2D()(fe)
  # output layer nodes
  fe = WeightNormalization(Dense(n_classes))(fe)
  
  # supervised output
  c_out_layer = Activation('softmax')(fe)
  # define and compile supervised discriminator model
  supervised_model = Model(in_image, c_out_layer)
  supervised_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer_grad, metrics=['accuracy'])
  
  # unsupervised output
  d_out_layer = Lambda(custom_activation)(fe)
  # define and compile unsupervised discriminator model
  unsupervised_model = Model(in_image, d_out_layer)
  unsupervised_model.compile(loss='binary_crossentropy', optimizer=optimizer_grad)
  
  return unsupervised_model, supervised_model



# # Discriminator Model Class Definition
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # Block 1: input is (3) x 64 x 64
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Block 2: input is (64) x 32 x 32
#             nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Block 3: input is (64*2) x 16 x 16
#             nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Block 4: input is (64*4) x 8 x 8
#             nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Block 5: input is (64*8) x 4 x 4
#             nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid(),
#             nn.Flatten()
#             # Output: 1
#         )

#     def forward(self, input):
#         out = self.main(input)
#         return out.view(-1, 1).squeeze(1)
