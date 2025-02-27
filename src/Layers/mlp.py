import numpy as np
from typing import Tuple, Literal
import matplotlib.pyplot as plt

from src.DataLoaders import DataLoader
from src.Layers import Layer
from src.Functions import LossFunction, Softmax, CrossEntropy


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)

        :param x: network input
        :return: network output
        """
        if self.layers is None or len(self.layers) == 0:
            raise ValueError("No layers defined.")

        prev_output = X
        for layer in self.layers:
            prev_output = layer.forward(prev_output)
        YHat = prev_output
        return YHat

    def backward(self, loss_function: LossFunction, Y: np.ndarray, YHat: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_function: function used to compute loss and its derivative
        :param Y: Y samples for batch
        :param YHat: forward-generated YHat samples corresponding to Y
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """

        dL_dW, dL_dB = [], []
        delta = dL_dYHat = loss_function.derivative(Y, YHat)

        for layer in reversed(self.layers):
            # Special case for last layer softmax activation function and cross entropy loss combo
            # Significant enough to make special case (numerical stability, etc.)
            soft_cross_special_case_dL_dz = None
            if layer == self.layers[-1] and isinstance(layer.activation_function, Softmax) and isinstance(loss_function, CrossEntropy):
                soft_cross_special_case_dL_dz = YHat - Y

            # General case
            dL_dW_l, dL_dB_l = layer.backward(delta, soft_cross_special_case_dL_dz)
            dL_dW.append(dL_dW_l)
            dL_dB.append(dL_dB_l)

            delta = np.matmul(layer.dL_dz, layer.W.T)

        dL_dW.reverse()
        dL_dB.reverse()
        return dL_dW, dL_dB

    def train(self, data_loader: DataLoader, loss_function: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param data_loader: DataLoader instance which enables batch generation for train, validate, and test sets.
        :param loss_function: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """

        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            train_generator = data_loader.batch_generator(batch_size, mode="train")
            validate_generator = data_loader.batch_generator(batch_size, mode="validate")

            training_loss = None
            validation_loss = None

            for (X_train, Y_train), (X_val, Y_val) in data_loader.zip_generators(train_generator, validate_generator):
                # Forward and backward
                YHat = self.forward(X_train)
                dL_dW, dL_dB = self.backward(loss_function, Y_train, YHat)

                # Update weights and biases
                for layer, (dL_dW_l, dL_dB_l) in zip(self.layers, zip(dL_dW, dL_dB)):
                    layer.W -= learning_rate * dL_dW_l
                    layer.b -= learning_rate * dL_dB_l

                # Training loss for chart
                if training_loss is None:
                    training_loss = 0
                training_loss += loss_function.loss(Y_train, YHat).sum()

                # Validation loss for chart (if exists this batch)
                if (X_val is None) or (Y_val is None):
                    continue

                YHat = self.forward(X_val)
                if validation_loss is None:
                    validation_loss = 0
                validation_loss += loss_function.loss(Y_val, YHat).sum()

            # Run through rest of validation batches, if exist (requirement of assignment)
            for X_val, Y_val in validate_generator:
                YHat = self.forward(X_val)
                validation_loss += loss_function.loss(Y_val, YHat)

            # Average losses
            training_loss /= data_loader.n_training_batches
            validation_loss /= data_loader.n_validation_batches

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            print(f"Epoch {epoch + 1}/{epochs} - \tTraining Loss: {training_loss:.8f}\t\t|\t\tValidation Loss: {validation_loss:.8f}")

        self.training_losses = np.array(training_losses)
        self.validation_losses = np.array(validation_losses)

        return self.training_losses, self.validation_losses

    def _test_classification(self, data_loader: DataLoader) -> np.ndarray:
        """
        Test the multilayer perceptron for classification
        """
        total_correct = 0
        total = 0

        test_generator = data_loader.batch_generator(batch_size=1, mode="test")

        for X_test, Y_test in test_generator:
            YHat = self.forward(X_test)
            predicted_class = np.argmax(YHat, axis=1)
            correct_class = np.argmax(Y_test, axis=1)

            if predicted_class == correct_class:
                total_correct += 1
            total += 1

        return total_correct * 1.0 / total

    def _test_regression(self, data_loader: DataLoader, denormalizer = None) -> np.ndarray:
        total_loss = 0
        n = 0

        test_generator = data_loader.batch_generator(batch_size=1, mode="test")

        for X_test, Y_test in test_generator:
            YHat = self.forward(X_test)
            if denormalizer:
                YHat = denormalizer(YHat)
            total_loss += np.sum((Y_test - YHat) ** 2)
            n += 1

        return np.sqrt(total_loss / n)

    def test(self, data_loader: DataLoader, mode: Literal['classification', 'regression'], denormalizer=None) -> np.ndarray:
        if mode == "classification":
            return self._test_classification(data_loader)
        elif mode == "regression":
            return self._test_regression(data_loader, denormalizer=denormalizer)
        else:
            raise ValueError("Invalid mode. Must be 'classification' or 'regression'.")

    def graph_training_losses(self):
        training = self.training_losses
        validating = self.validation_losses
        plt.plot(range(len(training)), training, label="Training Loss")
        plt.plot(range(len(validating)), validating, label="Validation Loss",
                 linestyle='dashed', linewidth=0.5)
        plt.legend(
            loc='upper right'
        )
        plt.show()