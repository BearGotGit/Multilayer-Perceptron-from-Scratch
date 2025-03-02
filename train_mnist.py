import numpy as np

from src.DataLoaders import DataLoader
from src.Functions import Relu, SquaredError, Sigmoid, MeanSquaredError, Mish, SoftPlus, CrossEntropy, Softmax
from src.Layers import Layer, MultilayerPerceptron

from data.mnist import MNIST_TRAIN_FEATURES, MNIST_TRAIN_LABELS, MNIST_TEST_FEATURES, MNIST_TEST_LABELS, \
    TITLES_2_COMPARE, IMAGES_2_TEST, show_images

seed = 69

# Architecture

h1 = Layer(784, 256, Relu(),
           dropout_probability=0.3,
           seed=seed)
h1_half = Layer(256, 256, Relu(),
                dropout_probability=0.3,
                seed=seed)
h2 = Layer(256, 128, Relu(),
           dropout_probability=0.3,
           seed=seed)
h2_half = Layer(128, 128, Relu(),
                dropout_probability=0.3,
                seed=seed)
h3 = Layer(128, 64, Sigmoid(), seed=seed)

h4 = Layer(64, 64, Relu(),
           dropout_probability=0.3,
           seed=seed)
h4_half = Layer(64, 32, Relu(),
                dropout_probability=0.3,
                seed=seed)
h5 = Layer(32, 32, Relu(),
           dropout_probability=0.3,
           seed=seed)
h5_half = Layer(32, 16, Relu(),
                dropout_probability=0.3,
                seed=seed)
h6 = Layer(16, 16, Relu(), seed=seed)

out = Layer(16, 10, Softmax(), seed=seed)

mlp = MultilayerPerceptron((h1, h1_half, h2, h2_half, h3, h4 , h4_half, h5 , h5_half, h6, out), seed=seed)

# Data

mnist_dataloader = DataLoader(MNIST_TRAIN_FEATURES, MNIST_TRAIN_LABELS, 80, 20, 0, seed)
mnist_test_dataloader = DataLoader(MNIST_TEST_FEATURES, MNIST_TEST_LABELS, 0, 0, 100, seed)

# Train

mlp.train(mnist_dataloader, CrossEntropy(), learning_rate=1E-2, batch_size=18, epochs=15, dropout=True)

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