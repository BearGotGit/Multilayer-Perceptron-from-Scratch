from src.DataLoaders import DataLoader
from src.Functions import Softmax, Relu, Linear, SquaredError, CrossEntropy, Sigmoid, MeanSquaredError, Mish, SoftPlus
from Layers import Layer, MultilayerPerceptron

from data.mnist import MNIST_TRAIN_FEATURES, MNIST_TRAIN_LABELS, MNIST_TEST_FEATURES, MNIST_TEST_LABELS

seed = 69

# Architecture

h1 = Layer(784, 64, Relu(), seed)
h2 = Layer(64, 32, Relu(), seed)
h3 = Layer(32, 20, SoftPlus(), seed)
h4 = Layer(20, 10, Sigmoid(), seed)

mlp = MultilayerPerceptron((h1, h2, h3, h4))

# Data

mnist_dataloader = DataLoader(MNIST_TRAIN_FEATURES, MNIST_TRAIN_LABELS, 8, 1, 1, seed)
mnist_test_dataloader = DataLoader(MNIST_TEST_FEATURES, MNIST_TEST_LABELS, 0, 0, 1, seed)

# Train

mlp.train(mnist_dataloader, SquaredError(), learning_rate=1E-3, batch_size=32, epochs=15, chart_batch=False)

# Graph

mlp.graph_training_losses()

# Test Accuracy

test_accuracy = mlp.test(mnist_test_dataloader)
print('(MNIST) Final accuracy on test set: ', test_accuracy)