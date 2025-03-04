import numpy as np

from src.DataLoaders import DataLoader
from src.Functions import Relu, SquaredError, Sigmoid, MeanSquaredError, Mish, SoftPlus, CrossEntropy, Softmax
from src.Layers import Layer, MultilayerPerceptron

from data.mnist import MNIST_TRAIN_FEATURES, MNIST_TRAIN_LABELS, MNIST_TEST_FEATURES, MNIST_TEST_LABELS, \
    TITLES_2_COMPARE, IMAGES_2_TEST, show_images

seed = 69

# Deep Architecture with about twice as many layers
h1 = Layer(784, 512, Sigmoid(), dropout_probability=0.3, seed=seed)
h2 = Layer(512, 256, Sigmoid(), dropout_probability=0.3, seed=seed)
h3 = Layer(256, 256, Sigmoid(), dropout_probability=0.3, seed=seed)
h4 = Layer(256, 128, Sigmoid(), dropout_probability=0.3, seed=seed)
h5 = Layer(128, 128, Sigmoid(), dropout_probability=0.3, seed=seed)
h6 = Layer(128, 64, Sigmoid(), dropout_probability=0.3, seed=seed)
h7 = Layer(64, 64, Sigmoid(), dropout_probability=0.3, seed=seed)
h8 = Layer(64, 64, Sigmoid(), dropout_probability=0.3, seed=seed)

# Building up again (widening)
h9  = Layer(64, 128, Sigmoid(), dropout_probability=0.3, seed=seed)
h10 = Layer(128, 128, Sigmoid(), dropout_probability=0.3, seed=seed)
h11 = Layer(128, 256, Sigmoid(), dropout_probability=0.3, seed=seed)
h12 = Layer(256, 256, Sigmoid(), dropout_probability=0.3, seed=seed)
h13 = Layer(256, 512, Sigmoid(), dropout_probability=0.3, seed=seed)

# Compressing towards the output
h14 = Layer(512, 256, Sigmoid(), dropout_probability=0.3, seed=seed)
h15 = Layer(256, 128, Sigmoid(), dropout_probability=0.3, seed=seed)
h16 = Layer(128, 64, Sigmoid(), dropout_probability=0.3, seed=seed)
out = Layer(64, 10, Softmax(), seed=seed)

mlp = MultilayerPerceptron((h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, out), seed=seed)

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