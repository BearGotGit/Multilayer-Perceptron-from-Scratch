import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

try:
    import torch
except ModuleNotFoundError:
    import unittest
    raise unittest.SkipTest("Skipping test_torch_mlp: torch not installed")

import numpy as np

# Updated imports from src.
from src.torch_mlp import Sigmoid as T_Sigmoid, Linear as T_Linear, Layer as T_Layer, MultilayerPerceptron as T_MLP, SquaredError as T_SquaredError
from src.Functions.activations import Sigmoid as NP_Sigmoid, Linear as NP_Linear
from src.Functions.losses import SquaredError as NP_SquaredError, MeanSquaredError as NP_MeanSquaredError
from src.Layers.layer import Layer as NP_Layer
from src.Layers.mlp import MultilayerPerceptron as NP_MLP

# Set common seeds for reproducibility
np_seed = 123
torch_seed = 123
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

# Define architecture parameters
input_dim = 3
hidden_dim = 4
output_dim = 2

# Compute fixed initial weights and biases for layer1 and layer2
def init_params(fan_in, fan_out):
    glorot_stddev = np.sqrt(6 / (fan_in + fan_out))
    W = np.random.normal(0, glorot_stddev, (fan_in, fan_out))
    b = np.ones(fan_out)
    return W, b

W1, b1 = init_params(input_dim, hidden_dim)
W2, b2 = init_params(hidden_dim, output_dim)

# Create a small synthetic dataset (10 samples)
num_samples = 10
X_np = np.random.randn(num_samples, input_dim)
# For testing, let targets be a linear function of inputs shifted
Y_np = X_np.dot(np.random.randn(input_dim, output_dim)) + 0.5

# ====== Setup numpy MLP ======
# Create two layers with numpy activations
np_layer1 = NP_Layer(input_dim, hidden_dim, NP_Sigmoid())
np_layer2 = NP_Layer(hidden_dim, output_dim, NP_Linear())
np_mlp = NP_MLP((np_layer1, np_layer2))
np_loss_func = NP_MeanSquaredError()

# Override weights and biases with fixed values
np_layer1.W = W1.copy()
np_layer1.b = b1.copy()
np_layer2.W = W2.copy()
np_layer2.b = b2.copy()

# Perform forward pass for numpy network
Y_pred_np = np_mlp.forward(X_np)
# Compute loss (mean squared error)
loss_np = np_loss_func.loss(Y_np, Y_pred_np)
# Compute derivative of loss wrt outputs
dL_dO_np = np_loss_func.derivative(Y_np, Y_pred_np)
# Run one backpropagation round to get gradients
np_grads_W, np_grads_b = np_mlp.backward(dL_dO_np, X_np)

# ====== Setup torch MLP ======
# Convert dataset to torch tensors
X_torch = torch.tensor(X_np, dtype=torch.float32)
Y_torch = torch.tensor(Y_np, dtype=torch.float32)

# Create torch layers with matching architecture
t_layer1 = T_Layer(input_dim, hidden_dim, T_Sigmoid())
t_layer2 = T_Layer(hidden_dim, output_dim, T_Linear())
t_mlp = T_MLP([t_layer1, t_layer2])
t_loss_func = T_SquaredError()

# Override weights and biases with same fixed values (convert to torch tensor)
with torch.no_grad():
    t_layer1.W.copy_(torch.tensor(W1, dtype=torch.float32))
    t_layer1.b.copy_(torch.tensor(b1, dtype=torch.float32))
    t_layer2.W.copy_(torch.tensor(W2, dtype=torch.float32))
    t_layer2.b.copy_(torch.tensor(b2, dtype=torch.float32))

# Zero gradients
for layer in t_mlp.layers:
    if layer.W.grad is not None:
        layer.W.grad.zero_()
    if layer.b.grad is not None:
        layer.b.grad.zero_()

# Forward pass for torch network
Y_pred_torch = t_mlp.forward(X_torch)
# Compute loss (MSE) (sum allows gradients to propagate correctly)
loss_torch = t_loss_func.loss(Y_torch, Y_pred_torch).mean()
# Backward pass
loss_torch.backward()

# ====== Verify Consistency of Architectures ======
print("\n\nVerifying architecture consistency:")
for i, (np_layer, t_layer) in enumerate(zip(np_mlp.layers, t_mlp.layers), start=1):
    np_w = np_layer.W
    np_b = np_layer.b
    t_w = t_layer.W.detach().numpy()
    t_b = t_layer.b.detach().numpy()
    print(f"Layer {i} weight difference (architecture):", np.linalg.norm(np_w - t_w))
    print(f"Layer {i} bias difference (architecture):", np.linalg.norm(np_b - t_b))

# ====== Compare Activation Functions ======
print("\nVerifying activation function consistency:")

# Import additional activation functions from numpy and torch implementations
from src.Functions.activations import Sigmoid as NP_Sigmoid, Tanh as NP_Tanh, Relu as NP_Relu, Softmax as NP_Softmax, Linear as NP_Linear
from src.torch_mlp import Sigmoid as T_Sigmoid, Tanh as T_Tanh, Relu as T_Relu, Softmax as T_Softmax, Linear as T_Linear

test_input_np = np.array([[0.5, -1.0, 2.0]])
test_input_torch = torch.tensor(test_input_np, dtype=torch.float32)

activations = [
    ("Sigmoid", NP_Sigmoid(), T_Sigmoid()),
    ("Tanh", NP_Tanh(), T_Tanh()),
    ("Relu", NP_Relu(), T_Relu()),
    ("Softmax", NP_Softmax(), T_Softmax()),
    ("Linear", NP_Linear(), T_Linear())
]

for name, np_act, t_act in activations:
    np_out = np_act.forward(test_input_np)
    t_out = t_act.forward(test_input_torch).detach().numpy()
    diff = np.linalg.norm(np_out - t_out)
    print(f"{name} output difference:", diff)

# ====== Compare gradients ======
print("\n\nComparing gradients for each layer:")

print('\n\n')
print('np_grads_W: ', np_grads_W)
print('np_grads_b: ', np_grads_b)
print('\n\n')

for i, (np_layer, t_layer) in enumerate(zip(np_mlp.layers, t_mlp.layers), start=1):
    print(f"Layer {i}:")
    print("Numpy W gradient:")
    print(np_grads_W[i-1])
    print("Torch W gradient:")
    print(t_layer.W.grad.detach().numpy())
    print("Difference (W):", np.linalg.norm(np_grads_W[i-1] - t_layer.W.grad.detach().numpy()))
    print("Diff (Factor) (W): ", (np_grads_W[i-1] / t_layer.W.grad.detach().numpy()))
    
    print("Numpy b gradient:")
    print(np_grads_b[i-1])
    print("Torch b gradient:")
    print(t_layer.b.grad.detach().numpy())
    print("Difference (b):", np.linalg.norm(np_grads_b[i-1] - t_layer.b.grad.detach().numpy()))
    print("Diff (Factor) (b): ", (np_grads_b[i-1] / t_layer.b.grad.detach().numpy()))

    print("-" * 60)
