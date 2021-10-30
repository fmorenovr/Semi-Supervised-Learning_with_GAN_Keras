from keras import backend as K

# Custom Activation for the Unsupervised model

# custom activation function
def custom_activation(output):
    logexpsum = K.sum(K.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result
