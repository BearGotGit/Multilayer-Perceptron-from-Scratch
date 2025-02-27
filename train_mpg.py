from src.DataLoaders import DataLoader
from src.Functions import Tanh, Relu, Linear, SquaredError, Sigmoid, MeanSquaredError
from src.Layers import Layer, MultilayerPerceptron

from data.mpg_dataset import AUTO_MPG_NORMALIZED_FEATURES, AUTO_MLP_NORMALIZED_LABELS

seed = 420

# Architecture

# h1 = Layer(7, 6, Linear(), seed)
# h2 = Layer(6, 5, Linear(), seed)
# h3 = Layer(5, 4, Linear(), seed)
# h4 = Layer(4, 3, Linear(), seed)
# h5 = Layer(3, 2, Linear(), seed)
# out = Layer(2, 1, Linear(), seed)

h1 = Layer(7, 7, Sigmoid(), seed)
out = Layer(7, 1, Linear(), seed)

# h3 = Layer(7, 5, Sigmoid(), seed)
# h4 = Layer(5, 4, Linear(), seed)
# h5 = Layer(4, 3, Linear(), seed)
# out = Layer(3, 1, Linear(), seed)

# mlp = MultilayerPerceptron((h1, h2, h3, h4, h5, out))
# mlp = MultilayerPerceptron((h1, h2, h3, h4, h5, out))
mlp = MultilayerPerceptron((h1, out))

# Data

mpg_dataloader = DataLoader(AUTO_MPG_NORMALIZED_FEATURES, AUTO_MLP_NORMALIZED_LABELS, 70, 15, 15, seed)

# Train

mlp.train(mpg_dataloader, SquaredError(), learning_rate=1E-3, batch_size=10, epochs=200)

# Graph

mlp.graph_training_losses()

# Test Accuracy

test_rmse = mlp.test(mpg_dataloader, mode="regression")
print(f'(MPG) Final error (RMSE) on test set: {test_rmse}.')
