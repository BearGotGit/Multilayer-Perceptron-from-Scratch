import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from src.DataLoaders import DataLoader
from src.Layers import Layer
from src.Functions import LossFunction, MeanSquaredError, Softmax, CrossEntropy


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

        #  General implementation based on:
        # O1 = h1.forward(X)
        # O2 = h2.forward(O1)
        # YHat = O3 = h3.forward(O2)

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

    def train(self, data_loader: DataLoader, loss_function: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, chart_batch: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_function: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """

        # Check for shape of train_x, train_y ... Make sure wouldn't break in architecture

        training_losses = []
        validation_losses = []

        # For all epochs
        for epoch in range(epochs):

            train_generator = data_loader.batch_generator(batch_size, mode="train")
            validate_generator = data_loader.batch_generator(batch_size, mode="validate")

            training_loss = None
            validation_loss = None

            # For each batch in an epoch
            for (X_train, Y_train), (X_val, Y_val) in data_loader.zip_generators(train_generator, validate_generator):

                # Forward and backward
                YHat = self.forward(X_train)
                dL_dW, dL_dB = self.backward(loss_function, Y_train, YHat)

                # Update weights and biases
                for layer, (dL_dW_l, dL_dB_l) in zip(self.layers, zip(dL_dW, dL_dB)):
                    layer.W -= learning_rate * dL_dW_l
                    layer.b -= learning_rate * dL_dB_l

                # Training loss for chart
                training_loss = loss_function.loss(Y_train, YHat).mean()
                if chart_batch:
                    training_losses.append(training_loss)

                # Validation loss for chart (if valid this batch)
                if (X_val is None) or (Y_val is None):
                    if chart_batch:
                        validation_losses.append(None)
                    continue

                YHat = self.forward(X_val)
                validation_loss = loss_function.loss(Y_val, YHat).mean()
                if chart_batch:
                    validation_losses.append(validation_loss)

            # Option to chart epoch, not batch
            if not chart_batch:
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)

        self.training_losses = training_losses
        self.validation_losses = validation_losses

        return self.training_losses, self.validation_losses

    def test(self, data_loader: DataLoader) -> np.ndarray:
        """
        Test the multilayer perceptron
        """
        total_correct = 0
        total = 0

        for X_test, Y_test in data_loader.batch_generator(batch_size=1, mode="test"):
            YHat = self.forward(X_test)
            predicted_class = np.argmax(YHat, axis=1)
            correct_class = np.argmax(Y_test, axis=1)

            if predicted_class == correct_class:
                total_correct += 1
            total += 1

        if total == 0:
            print("No no no.")
            return np.array(0)

        return total_correct * 1.0 / total

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