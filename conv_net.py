from numpy import float

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

NUM_CONV_FEATS = 32
KERNEL_SIZE = 3
POOL_SIZE = 2

P_DROP2 = 0.25
P_DROP3 = 0.5

NUM_FEATURES = 784
IMG_SIZE = 28
NUM_CLASSES = 10

def build_model(num_conv_feats=NUM_CONV_FEATS, kernel_size=KERNEL_SIZE,
                pool_size=POOL_SIZE, p_drop2=P_DROP2, p_drop3=P_DROP3):
    # Model is a sequential stack of layers
    model = Sequential()

    # Layer 1: convolution + ReLU
    model.add(Convolution2D(num_conv_feats, kernel_size, kernel_size,
                            border_mode='valid',
                            input_shape=(1, IMG_SIZE, IMG_SIZE),
                            name='Conv1'))
    model.add(Activation('relu', name='ReLU1'))

    # Layer 2: convolution + ReLU + pooling (with dropout)
    model.add(Convolution2D(num_conv_feats, kernel_size, kernel_size,
                            name='Conv2'))
    model.add(Activation('relu', name='ReLU2'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), name='Pool2'))
    # Keras is a bit temperamental regarding float precision
    model.add(Dropout(float(p_drop2), name='Drop2'))

    # Layer 3: normal layer + ReLU (with dropout)
    model.add(Flatten(name='Flat3'))
    model.add(Dense(128, input_dim=NUM_FEATURES, name='Dense3'))
    model.add(Activation('relu', name='ReLU3'))
    # Keras is a bit temperamental regarding float precision
    model.add(Dropout(float(p_drop3), name='Drop3'))

    # Layer 4: output + softmax
    model.add(Dense(NUM_CLASSES, name='Dense4'))
    model.add(Activation('softmax', name='Soft4'))

    return model
