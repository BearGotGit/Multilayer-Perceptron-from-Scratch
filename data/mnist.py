# L. Deng, "The MNIST Database of Handwritten Digit Images for Machine Learning Research [Best of the Web]," in IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 141-142, Nov. 2012, doi: 10.1109/MSP.2012.2211477.

# Most code copied:
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset?scriptVersionId=9466282&cellId=1
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset?scriptVersionId=9466282&cellId=2

import numpy as np
import struct
from array import array
from os.path import join


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.array(array("B", file.read()))

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.array(array("B", file.read()))

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)




# ============================================================================================================




import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
# (NOTE: This is called from src/from_scratch.py), so would need to update the data_directory if called from somewhere else...
#
data_directory = './data/mnist'
training_images_filepath = join(data_directory, 'train-images.idx3-ubyte')
training_labels_filepath = join(data_directory, 'train-labels.idx1-ubyte')
test_images_filepath = join(data_directory, 't10k-images.idx3-ubyte')
test_labels_filepath = join(data_directory, 't10k-labels.idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15)
        index += 1

    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(train_x, train_y), (test_x, test_y) = mnist_dataloader.load_data()

# Flatten for multilayer perceptron and Normalize in [0,1]
# One-hot encode the labels

MNIST_TRAIN_FEATURES = np.array(train_x).reshape(60000, 784) / 255.0
'''
60K training samples, 784 features per sample, normalized 0 to 1 (flattened 28x28 grey-scale image)
'''
MNIST_TRAIN_LABELS = np.eye(10)[train_y]
'''
60K training samples, 10 classes, one-hot encoded. [1 0 ... 0] for 0, [0 1 ... 0] for 1, etc.
'''
MNIST_TEST_FEATURES = np.array(test_x).reshape(10000, 784) / 255.0
'''
10K training samples, 784 features per sample, normalized 0 to 1 (flattened 28x28 grey-scale image)
'''
MNIST_TEST_LABELS = np.eye(10)[test_y]
'''
10K test samples, 10 classes, one-hot encoded. [1 0 ... 0] for 0, [0 1 ... 0] for 1, etc.
'''

print("MNIST dataset loaded.")


IMAGES_2_TEST = []
TITLES_2_COMPARE = []

for i in range(0, 10):
    r = list(test_y[:100]).index(i)
    IMAGES_2_TEST.append(test_x[r])
    TITLES_2_COMPARE.append('test image [' + str(r) + '] = ' + str(test_y[r]))

show_images(IMAGES_2_TEST, TITLES_2_COMPARE)

