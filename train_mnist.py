import numpy as np

from src.DataLoaders import DataLoader
from src.Functions import Relu, SquaredError, Sigmoid, MeanSquaredError, Mish, SoftPlus, CrossEntropy, Softmax
from src.Layers import Layer, MultilayerPerceptron

from data.mnist import MNIST_TRAIN_FEATURES, MNIST_TRAIN_LABELS, MNIST_TEST_FEATURES, MNIST_TEST_LABELS, \
    TITLES_2_COMPARE, IMAGES_2_TEST, show_images

seed = 69

# Architecture

h1 = Layer(784, 64, Relu(), seed)
h2 = Layer(64, 32, Relu(), seed)
h3 = Layer(32, 20, Relu(), seed)
out = Layer(20, 10, Softmax(), seed)

mlp = MultilayerPerceptron((h1, h2, h3, out))

# Data

mnist_dataloader = DataLoader(MNIST_TRAIN_FEATURES, MNIST_TRAIN_LABELS, 80, 20, 0, seed)
mnist_test_dataloader = DataLoader(MNIST_TEST_FEATURES, MNIST_TEST_LABELS, 0, 0, 100, seed)

# Train

mlp.train(mnist_dataloader, CrossEntropy(), rmsprop_beta=0.95, rmsprop_epsilon=10e-8, learning_rate=1E-3, batch_size=32, epochs=15)

# Graph

mlp.graph_training_losses()

# Test Accuracy

test_accuracy = mlp.test(mnist_test_dataloader, mode="classification")
print(f'(MNIST) Final accuracy on test set: {test_accuracy * 100}%')

# Comparison Table

# for each number in ten_numbers, predict the number
# ... modify corresponding title to include prediction
# ... show_images(ten_numbers, TITLES_2_COMPARE)
for i in range(len(IMAGES_2_TEST)):
    image = np.array(IMAGES_2_TEST[i]) / 255.0
    n = image.reshape(-1, 784)
    prediction = mlp.forward(n)

    TITLES_2_COMPARE[i] += f' | prediction: {np.argmax(prediction)}'

show_images(IMAGES_2_TEST, TITLES_2_COMPARE)