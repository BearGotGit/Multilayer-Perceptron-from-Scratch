# Foundational Models Project 1

This project implements a Multilayer Perceptron (MLP) and includes various activation and loss functions. The project also includes a DataLoader class for handling dataset splitting and batch generation.

## Environment Setup

1. **Install Conda**: If you haven't installed Conda yet, follow the instructions on the [official Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create and Activate Conda Environment**:

   ```sh
   conda create --name berend_grandt_foundational_ai_project_1 python=3.11
   conda activate berend_grandt_foundational_ai_project_1
   ```

3. **Install Dependencies**: #FIXME: This is actually not right. Update when done:
   ```sh
   conda install --file requirements.txt
   ```

## Project Structure

```
.
├── src
│   ├── __init__.py           # Package initialization
│   ├── activation_functions.py # Implementation of activation functions
│   ├── data_loader.py        # Implementation of DataLoader class
│   ├── loss_functions.py     # Implementation of loss functions
│   └── mlp.py                # Implementation of MLP and related classes
├── testing
│   ├── __init__.py           # Package initialization
│   ├── test_torch_mlp.py     # Testing against torch implementation
│   └── test_mlp.py           # Unit tests for the MLP implementation
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```

## Running Tests

To run the tests and check the coverage, use the following commands:

```sh
coverage run --source=src -m unittest discover -s testing -p "*.py" -v
coverage report -m
```

To see the report as an HTML page, use this command and open the generated url:

```sh
coverage html
```

## Usage

### DataLoader

The `DataLoader` class is used to split the dataset into training, validation, and test sets and generate batches for training.

### Activation Functions

The project includes the following activation functions:

- Sigmoid
- Tanh
- Relu
- Softmax
- Linear

### Loss Functions

The project includes the following loss functions:

- SquaredError
- CrossEntropy

### Multilayer Perceptron

The `MultilayerPerceptron` class is used to create and train a multilayer perceptron.

## Example

Here is an example of how to use the `MultilayerPerceptron` class:

```python
from src.mlp import DataLoader, Sigmoid, Layer, MultilayerPerceptron, SquaredError

# Create a DataLoader instance
data_loader = DataLoader((train_x, train_y), num_train=100, num_valid=20, num_test=20, seed=42)

# Define the layers
layer1 = Layer(2, 2, Sigmoid())
layer2 = Layer(2, 1, Sigmoid())

# Create the MLP
mlp = MultilayerPerceptron((layer1, layer2))

# Train the MLP
training_losses, validation_losses = mlp.train(train_x, train_y, val_x, val_y, SquaredError(), learning_rate=0.001, batch_size=16, epochs=32)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
