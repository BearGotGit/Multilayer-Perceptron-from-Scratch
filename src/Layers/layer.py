import numpy as np
from typing import Tuple
from src.Functions import ActivationFunction, Softmax


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, seed: int = 69):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynapatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        np.random.seed(seed)

        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # Initialize weights and biases
        glorot_stddev = np.sqrt(2 / (fan_in + fan_out) )
        self.W = np.random.normal(0, glorot_stddev, (fan_in, fan_out))
        self.b = np.zeros([1, fan_out])

        # Initialize RMSProp variables
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        # Forward only
        z = np.matmul(h, self.W) + self.b
        O = self.activation_function.forward(z)

        # Useful for backprop
        self.h = h
        self.dO_dz = self.activation_function.derivative(z)
        # n samples * identity like bias axis 0
        self.dz_db = len(h) * np.identity(self.fan_out)

        # Return forward
        return O

    def backward(self, delta: np.ndarray, dL_dz_softmax_and_crossentropy: np.ndarray = None, rmsprop_beta = None, rmsprop_epsilon = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param delta: delta term from layer above
        :param dL_dz_softmax_and_crossentropy: special case for softmax and cross entropy loss function
        :param rmsprop_beta: RMSProp decay rate
        :param rmsprop_epsilon: small value to avoid division by zero
        :return: (weight gradients, bias gradients)
        """

        # Using softmax and cross entropy loss function together has simple dL_dz that's computed up one function call.
        if dL_dz_softmax_and_crossentropy is not None:
            self.dL_dz = dL_dz_softmax_and_crossentropy
        # Only Softmax example (let final layer be 10 cols, let 2nd final layer be 16):
        #   If softmax is activation function for layer,
        #   dO_dz is n x 10 x 10, rather than usual n x 10
        #   In special case, want to treat delta as n x (1 x 10), rather than n x 10,
        #   then collapse back to n x 10, so rest of backprop works.
        #   We can use einsum to represent this logic more concisely.
        elif isinstance(self.activation_function, Softmax):
            self.dL_dz = np.einsum("bj, bjk -> bk", delta, self.dO_dz)
        else:
            self.dL_dz = np.multiply(delta, self.dO_dz)

        dL_dW = np.matmul(self.h.T, self.dL_dz)
        dL_dB = np.matmul(self.dL_dz, self.dz_db)
        dL_dB = np.sum(dL_dB, axis=0, keepdims=True)

        if rmsprop_beta is not None and rmsprop_epsilon is not None:
            # Update RMSProp variables
            self.v_W = rmsprop_beta * self.v_W + (1 - rmsprop_beta) * (dL_dW ** 2)
            self.v_b = rmsprop_beta * self.v_b + (1 - rmsprop_beta) * (dL_dB ** 2)

            # Adjust gradients if RMSProp
            dL_dW = dL_dW / (np.sqrt(self.v_W) + rmsprop_epsilon)
            dL_dB = dL_dB / (np.sqrt(self.v_b) + rmsprop_epsilon)

        return dL_dW, dL_dB