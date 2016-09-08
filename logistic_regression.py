from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2

NUM_FEATURES = 784
NUM_CLASSES = 10

L2_COEF = 0.01

def build_model(l2_coef=L2_COEF):

    # Model is a sequential stack of layers
    model = Sequential()

    # Layer: output dense layer + softmax
    model.add(Dense(NUM_CLASSES, input_dim=NUM_FEATURES,
                    W_regularizer=l2(l2_coef), name='Output'))

    model.add(Activation('softmax', name='Soft'))

    return model
