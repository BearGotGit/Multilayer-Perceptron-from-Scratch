import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Note: Derivatives are taken with respect to the activations of y_pred, NOT the pre-activations of y_pred.
        """
        pass

class SquaredError(LossFunction):
    def loss(self, y_true, y_pred) -> np.ndarray:
        diff = (y_true - y_pred)
        result = 0.5 * diff ** 2
        return result
    
    def derivative(self, y_true, y_pred) -> np.ndarray:
        return y_pred - y_true

class MeanSquaredError(SquaredError):
    def loss(self, y_true, y_pred) -> np.ndarray:
        return super().loss(y_true, y_pred).mean(axis=1)

    def derivative(self, y_true, y_pred) -> np.ndarray:
        y_dim = len(y_true[0])
        return 1.0 / y_dim * super().derivative(y_true, y_pred)

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        # log(x) is actually ln(x) here
        inner = np.multiply(-y_true, np.log(y_pred))
        return np.sum(inner, axis=1)

    def derivative(self, y_true, y_pred):
        EPSILON = 1e-4

        y_pred = np.where(y_pred == 0, EPSILON, y_pred)
        return -y_true / y_pred