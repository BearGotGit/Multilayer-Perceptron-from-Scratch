import torch
import torch.nn.functional as F
import math

# Activation Functions
class ActivationFunction:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Abstract interface
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

class Tanh(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

class Relu(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

class Softmax(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=1)

class Linear(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

# Loss Functions
class LossFunction:
    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Abstract interface
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return 0.5 * (y_true - y_pred) ** 2

class CrossEntropy(LossFunction):
    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Assumes y_true is one-hot encoded
        return -torch.sum(y_true * torch.log(y_pred + 1e-8), dim=1)

# Layer definition
class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # Glorot initialization
        glorot_stddev = math.sqrt(6 / (fan_in + fan_out))
        self.W = (torch.randn(fan_in, fan_out) * glorot_stddev).requires_grad_()
        self.b = torch.ones(fan_out, requires_grad=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h.mm(self.W) + self.b
        return self.activation_function.forward(z)

# Multilayer Perceptron using torch tensors
class MultilayerPerceptron:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train_model(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                    val_x: torch.Tensor, val_y: torch.Tensor, 
                    loss_func: LossFunction, learning_rate: float = 1E-3, epochs: int = 32):
        for epoch in range(epochs):
            # Zero gradients
            for layer in self.layers:
                if layer.W.grad is not None:
                    layer.W.grad.zero_()
                if layer.b.grad is not None:
                    layer.b.grad.zero_()
            
            # Forward pass
            y_pred = self.forward(train_x)
            loss = loss_func.loss(train_y, y_pred).mean()
            
            # Backward pass via autograd
            loss.backward()

            # Update parameters
            for layer in self.layers:
                with torch.no_grad():
                    layer.W -= learning_rate * layer.W.grad
                    layer.b -= learning_rate * layer.b.grad

            # Validation loss
            with torch.no_grad():
                val_pred = self.forward(val_x)
                val_loss = loss_func.loss(val_y, val_pred).mean()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")
