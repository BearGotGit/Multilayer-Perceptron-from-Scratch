import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray):
        sig = self.forward(x)
        return sig * (1 - sig)


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()

        self.sig = Sigmoid().forward
    
    def forward(self, x):
        return 2 * self.sig(2 * x) - 1
    def derivative(self, x):
        return 1 - (self.forward(x)) ** 2


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray):
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray):
        return np.where(x > 0, 1, 0)

class SoftPlus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return Sigmoid().forward(x)

class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(SoftPlus().forward(x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x) + np.tanh(x)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray):
        """
        Applies the softmax function to the input array.

        Parameters:
        x (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
        np.ndarray: Output array with the same shape as input, where each row 
                represents the softmax probabilities of the corresponding input row.
        """
        e_raise = np.exp(x)
        sum_term = np.sum(e_raise, axis=1).reshape(-1, 1)

        # broadcasts division of sum across each individual term in a sample, not just for each sample
        return e_raise / sum_term

    def single_sample_derivative(self, x: np.ndarray):
        # In case passed x as vector, it needs to be matrix
        s_x = self.forward(x.reshape(1, -1))
        diag = np.diag(s_x[0])

        return diag - np.matmul(s_x.T, s_x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.single_sample_derivative, 1, x)

class Linear(ActivationFunction):
    def forward(self, x):
        return x
    def derivative(self, x):
        return np.ones_like(x)
    