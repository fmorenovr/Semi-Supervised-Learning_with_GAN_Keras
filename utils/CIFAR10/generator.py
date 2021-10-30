
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation, ReLU
from keras.layers import GlobalAveragePooling2D, BatchNormalization

#from tensorflow_addons.layers import WeightNormalization

# Defines a unsupersived output and supervised output.  
# Both shared the same weights until the last Dense.

# define the standalone generator model
def define_generator(latent_dim=100):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 512 * 4 * 4
    gen = Dense(n_nodes)(in_lat)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    # upsample 100 -> 4x4x512
    gen = Reshape((4, 4, 512))(gen)
    
    # upsample 4x4x512 -> 8x8x256
    gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding="same")(gen)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    
    # upsample 8x8x256 -> 16x16x128
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    
    # upsample 16x16x128 -> 32x32x64
    #gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(gen)
    #gen = BatchNormalization()(gen)
    #gen = ReLU()(gen)
    
    # upsample 32x32x64 -> 32x32x3
    #gen = Conv2D(3, (4,4), strides=(1,1), padding='same')(gen)
    gen = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same')(gen)
    
    # outlayer
    out_layer = Activation('tanh')(gen)
    
    # define model
    generator_model = Model(in_lat, out_layer)
    return generator_model



# define the standalone generator model
def define_generator_paper(latent_dim=100):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 4x4 image
    n_nodes = 512 * 4 * 4
    gen = Dense(n_nodes)(in_lat)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    gen = Reshape((4, 4, 512))(gen)
    
    # upsample to 4x4 -> 8x8
    gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    
    # upsample to 8x8 -> 16x16
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    
    # upsample to 16x16 -> 32x32
    #gen = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same')(gen)
    gen = WeightNormalization(Conv2DTranspose(3, (4,4), strides=(2,2), padding='same'), data_init=False)(gen)
    
    # outlayer
    out_layer = Activation('tanh')(gen)
    
    # define model
    generator_model = Model(in_lat, out_layer)
    return generator_model



# class Generator(nn.Module):
#     def __init__(self, latent_vector_size = 100):
#         super(Generator, self).__init__()
        
#         self.deconv1 = nn.ConvTranspose2d( latent_vector_size, 64 * 8, 4, 1, 0, bias=False)
#         self.norm1 = nn.BatchNorm2d(64 * 8)
#         self.relu = nn.ReLU(True)
#         self.deconv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
#         self.norm2 = nn.BatchNorm2d(64 * 4)
#         self.deconv3 = nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False)
#         self.norm3 = nn.BatchNorm2d(64*2)
#         self.deconv4 = nn.ConvTranspose2d( 64*2 , 64, 4, 2, 1, bias=False)
#         self.norm4 = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d( 64 , 3, 4, 2, 1, bias=False)
#         self.tanh = nn.Tanh()
        
    
#     def decode(self, z):
#         z = self.relu(self.norm1(self.deconv1(z))) # b, 16, 5, 5
#         z = self.relu(self.norm2(self.deconv2(z))) # b, 8, 15, 15
#         z = self.relu(self.norm3(self.deconv3(z))) # b, 1, 28, 28
#         z = self.relu(self.norm4(self.deconv4(z))) # b, 1, 28, 28
#         z = self.tanh(self.deconv5(z))
#         return z

#     def forward(self, z):
#         return self.decode(z)
