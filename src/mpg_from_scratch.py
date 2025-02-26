from src.DataLoaders import DataLoader
from src.Functions import Softmax, Relu, Linear, SquaredError, CrossEntropy, Sigmoid, MeanSquaredError, Mish, SoftPlus
from Layers import Layer, MultilayerPerceptron

from data.mpg_dataset import AUTO_MPG_FEATURES, AUTO_MPG_LABELS

seed = 420

# Architecture

h1 = Layer(7, 4, Relu(), seed)
h2 = Layer(4, 2, Relu(), seed)
h3 = Layer(2, 1, Linear(), seed)

mlp = MultilayerPerceptron((h1, h2, h3))

# Data

mpg_dataloader = DataLoader(AUTO_MPG_FEATURES, AUTO_MPG_LABELS, 70, 15, 15, seed)

# Train

mlp.train(mpg_dataloader, SquaredError(), learning_rate=1E-3, batch_size=100, epochs=1, chart_batch=True)

# Graph

mlp.graph_training_losses()

# Test Accuracy

test_accuracy = mlp.test(mpg_dataloader)
print('(MPG) Final accuracy on test set: ', test_accuracy)
