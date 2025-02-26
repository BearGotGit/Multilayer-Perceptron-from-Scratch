import sys
import os
import unittest
import numpy as np
from matplotlib import pyplot as plt
# 
from src.Functions.activations import Sigmoid, Tanh, Relu, Softmax, Linear
from src.Functions.losses import SquaredError, CrossEntropy
from src.Layers.layer import Layer
from src.Layers.mlp import MultilayerPerceptron

# 
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# FIXME: Lol, like none of your tests even test. 

class TestMLP(unittest.TestCase):
    def test_forward(self):
        
        layers = (Layer(3, 2, Linear()), Layer(2, 2, Linear()), Layer(2, 3, Linear()))
        mlp = MultilayerPerceptron(layers)

        x = np.random.randn(5, 3)

        y_hat = mlp.forward(x)

        print(f'y_hat: {np.shape(y_hat)}')


class TestActivationFunctions(unittest.TestCase):
    def test_sigmoid(self):
        act = Sigmoid()
        z = np.arange(-4, 4, 0.1)
        # Add assertions for forward and derivative methods
        a = act.forward(z)

        print(f'Shapes match: {np.shape(z) == np.shape(a)}')

        # plt.plot(z, a)
        # plt.xlabel('z')
        # plt.ylabel('a = sigmoid(z)')

        # plt.show()


    def test_tanh(self):
        act = Tanh()
        z = np.arange(-4, 4, 0.1)
        # Add assertions for forward and derivative methods
        a = act.forward(z)

        print(f'Shapes match: {np.shape(z) == np.shape(a)}')

        # plt.plot(z, a)
        # plt.xlabel('z')
        # plt.ylabel('a = tanh(z)')

        # plt.show()

    def test_relu(self):
        act = Relu()
        z = np.arange(-4, 4, 0.1)
        # Add assertions for forward and derivative methods
        a = act.forward(z)

        print(f'Shapes match: {np.shape(z) == np.shape(a)}')

        # plt.plot(z, a)
        # plt.xlabel('z')
        # plt.ylabel('a = relu(z)')

        # plt.show()

    def test_softmax(self):
        act = Softmax()

        z = np.array([[0, 0.1, 0.05, 1.5, 5.2, 4.0, 1., .1, 4., 2], [0, -5, 0.05, 1.5, 5.2, 8.0, 1., .1, 4., 2]], dtype=np.float32)

        # Add assertions for forward and derivative methods
        a = act.forward(z)

        print(f'Softmax(z) => {np.shape(a)}')

        print(f'Shapes match: {np.shape(z) == np.shape(a)}')

        # plt.scatter(z, a)
        # plt.xlabel(f'z')
        # plt.ylabel('a = softmax(z)')
        # plt.show()

        print('Sum over a: ', np.sum(a, axis=1))

    def test_linear(self):
        act = Linear()
        z = np.arange(-4, 4, 0.1)
        # Add assertions for forward and derivative methods
        a = act.forward(z)

        print(f'Shapes match: {np.shape(z) == np.shape(a)}')

        # plt.scatter(z, a)
        # plt.xlabel(f'z')
        # plt.ylabel('a = linear(z)')
        # plt.show()


class TestLossFunctions(unittest.TestCase):
    def test_squared_error(self):
        loss = SquaredError()
        y_true = np.array([[1, 0],[1, 0]])
        y_pred = np.array([[0.8, 0.2],[0.8, 0.2]])
        # Add assertions for loss and derivative methods

        l = loss.loss(y_true=y_true, y_pred=y_pred)

        print(f"loss = {l} \nshape: {np.shape(l)}")

    def test_cross_entropy(self):
        loss = CrossEntropy()
        y_true = np.array([[1, 0]])
        y_pred = np.array([[0.8, 0.2]])
        
        l = loss.loss(y_true=y_true, y_pred=y_pred)

        print(f"loss = {l}")
        print(f"shape = {np.shape(l)}")

        # Add assertions for loss and derivative methods

class TestLayer(unittest.TestCase):
    def test_forward(self):
        layer = Layer(3, 2, Sigmoid())
        h = np.array([[1, 2, 3], [-1, -2, -3], [-1, -2, -3], [-1, -2, -3]])

        yhat = layer.forward(h)
        self.assertEqual(np.shape(yhat), (4, 2))


    def test_backward_raises(self):
        layer = Layer(3, 2, Sigmoid())
        h = np.array([[1, 2, 3], [0, 0, 0]])
        
        # Note: no forward pass
        loss = np.array([[1, 2], [-2, -1]])

        with self.assertRaises(RuntimeError) as context:
            layer.backward(loss, prev_layer_O=h)
        self.assertEqual(str(context.exception), "Forward pass must occur before backward")


    def test_backward_single_layer(self):
        layer = Layer(2, 3, Sigmoid())
        h = np.array([[1, 2], [-1, -2]])

        _ = layer.forward(h)

        # loss is delta for single layer
        loss = np.array([[1, 2, 0], [-2, -1, 0]])
        dL_dW, dL_db = layer.backward(delta=loss, prev_layer_O=h)

        # print(layer)
        Oz = layer.dO_dz
        zW = layer.dz_dW
        zb = layer.dz_db

        # Eg: for dO_dz, 2 samples, dim is row of 3 gives (2, 3)
        self.assertEqual(np.shape(Oz), (2, 3))
        self.assertEqual(np.shape(zb), (3,))
        self.assertEqual(np.shape(zW), (2, 2))
        
        self.assertEqual(np.shape(dL_dW), (2, 3))
        self.assertEqual(np.shape(dL_db), (3,))


class TestMultilayerPerceptron(unittest.TestCase):
    def test_forward(self):
        layer1 = Layer(2, 2, Linear())
        layer2 = Layer(2, 1, Linear())
        mlp = MultilayerPerceptron((layer1, layer2))
        x = np.array([[1, 2]])
        x2 =  np.array([[1, 2], [1, 2]])

        # Add assertions for forward method

        y_hat = mlp.forward(x)

        print(f"yhat: {np.shape(y_hat)}")

        y_hat2 = mlp.forward(x2)

        print(f"yhat2: {np.shape(y_hat2)}")

        # For each layer, print shapes of the gradients

        for i, layer in enumerate(mlp.layers):
            print(f"{i+1}th layer:\n {layer}")

    def test_backward(self):
        f = 3
        q = 4
        n = 3

        layer1 = Layer(f, 2, Linear())
        layer2 = Layer(2, 2, Linear())
        layer3 = Layer(2, q, Linear())
        mlp = MultilayerPerceptron((layer1, layer2, layer3))

        x = np.array([[1, 2, 1], [1, 2, 1], [1, 2, 1]])
        
        yhat = mlp.forward(x)
        y = np.array([list(range(q))] * n)
        loss = SquaredError().loss(y, yhat)

        assert(np.shape(x) == (n, f))
        assert(np.shape(loss) == (n, q))
        
        dL_dW_all, dL_db_all = mlp.backward(loss_grad=loss, input_data=x)

        self.assertEqual(np.shape(dL_dW_all[0]), (f, 2))
        self.assertEqual(np.shape(dL_dW_all[1]), (2, 2))
        self.assertEqual(np.shape(dL_dW_all[2]), (2, q))

        self.assertEqual(np.shape(dL_db_all[0]), (2,))
        self.assertEqual(np.shape(dL_db_all[1]), (2,))
        self.assertEqual(np.shape(dL_db_all[2]), (q,))


    # def test_train(self):
    #     layer1 = Layer(2, 2, Linear())
    #     layer2 = Layer(2, 1, Linear())
    #     mlp = MultilayerPerceptron((layer1, layer2))
    #     train_x = np.array([[1, 2], [3, 4]])
    #     train_y = np.array([[1], [0]])
    #     val_x = np.array([[5, 6]])
    #     val_y = np.array([[1]])
    #     loss_func = SquaredError()
    #     # Add assertions for train method

if __name__ == '__main__':
    unittest.main()
