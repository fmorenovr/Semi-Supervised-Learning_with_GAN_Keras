
from keras.optimizers import Adam
from keras.models import Model

# We let the unsupervised model as not trainable, because we gonna trin the weights of the supervised model (which shares weights with the unsupervised).

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator_model, unsupervised_model, learning_rate = 0.0001):
  # make weights in the discriminator not trainable
  unsupervised_model.trainable = False
  
  # connect image output from generator as input to discriminator
  gan_output = unsupervised_model(generator_model.output)
  
  # define gan model as taking noise and outputting a classification
  model = Model(generator_model.input, gan_output)
  
  # optimizer
  optimizer_grad = Adam(lr=learning_rate, beta_1=0.5)
  
  # compile model
  model.compile(loss='binary_crossentropy', optimizer=optimizer_grad)
  return model
