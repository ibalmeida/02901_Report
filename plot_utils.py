import matplotlib.pyplot as plt
from numpy import reshape
from numpy.random import randint

IMG_SIZE = 28

def convert_to_image(image_vector):
    img = reshape(image_vector, (IMG_SIZE, IMG_SIZE), order='C')
    return img

def plot_pattern(image_vector, image_label=None):
    plt.figure()
    plt.imshow(convert_to_image(image_vector), cmap='Greys_r')
    if image_label:
        plt.title('True label:' + str(image_label))
    plt.show()

def plot_a_few(X, y):
    idx = randint(0, y.shape[0], 4)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    for i, ax in zip(idx, [ax1, ax2, ax3, ax4]):
        ax.set_title('True label:' + str(y[i]))
        ax.imshow(convert_to_image(X[i, :]), cmap='Greys_r')
