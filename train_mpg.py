import numpy as np

from src.DataLoaders import DataLoader
from src.Functions import Linear, SquaredError, Sigmoid
from src.Layers import Layer, MultilayerPerceptron

from data.mpg_dataset import AUTO_MPG_NORMALIZED_FEATURES, AUTO_MLP_NORMALIZED_LABELS, auto_mpg_denormalize_labels, \
    AUTO_MPG_LABELS

seed = 420

# Architecture

h1 = Layer(7, 7, Sigmoid(), seed)
out = Layer(7, 1, Linear(), seed)

mlp = MultilayerPerceptron((h1, out))

# Data

mpg_dataloader = DataLoader(AUTO_MPG_NORMALIZED_FEATURES, AUTO_MLP_NORMALIZED_LABELS, 70, 15, 15, seed)

# Train

mlp.train(mpg_dataloader, SquaredError(), learning_rate=1E-3, batch_size=10, epochs=200)

# Graph

mlp.graph_training_losses()

# Test Accuracy

denormalized_loader = DataLoader(AUTO_MPG_NORMALIZED_FEATURES, AUTO_MPG_LABELS, 0, 0, 100, seed)

test_rmse = mlp.test(denormalized_loader, mode="regression", denormalizer=auto_mpg_denormalize_labels)
print(f'(MPG) Final error (RMSE) on test set: {test_rmse}.')


# Comparison Table

table = np.zeros((10, 4))

# for 10 data points from test set, predict the miles per gallon
i = 10
print(f'[Prediction, Denormalized Predication, Theory, Denormalized Theory]')
for normalized_sample in mpg_dataloader.batch_generator(batch_size=1, mode="test"):
    if not i:
        break
    i -= 1
    #
    prediction = mlp.forward(normalized_sample[0])
    table[i] = [prediction.item(), auto_mpg_denormalize_labels(prediction).item(), normalized_sample[1].item(), auto_mpg_denormalize_labels(normalized_sample[1]).item()]
    # print(f'{prediction.item()},\t{auto_mpg_denormalize_labels(prediction).item()},\t{normalized_sample[1].item()},\t{auto_mpg_denormalize_labels(normalized_sample[1]).item()}')


print("\nMean of error (normalized): ", np.mean(table[:, 0] - table[:, 2]))
print("Mean of error (denormalized): ", np.mean(table[:, 1] - table[:, 3]))
print("Std dev of error (normalized): ", np.std(table[:, 0] - table[:, 2]))
print("Std dev of error (denormalized): ", np.std(table[:, 1] - table[:, 3]))