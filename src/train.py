import numpy as np
from matplotlib import pyplot as plt

# Obtain DataLoader, MLP model, Loss,
from src.Functions import MeanSquaredError

from src.DataLoaders import DataLoader

from src.Layers import Layer, MultilayerPerceptron
from src.Functions import Linear
from src.Functions import LossFunction


class Train:
    def __init__ (self, model: MultilayerPerceptron, loss_function: LossFunction, learning_rate: float,
            data_loader, optimizer: bool = True
            ):
        
        # 
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.data_loader = data_loader

        # 
        self.time_steps = None
        self.train_loss_history = None
        self.loss_history = None
        
        # Plot for training
        self.plot = plt.figure()
        ax = self.plot.gca()
        ax.set_title("Loss vs Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Loss")

    def train (self, epochs: int = 1):
        for _ in range(epochs):


            training_loader = self.data_loader.load_data(train=True)
            validation_loader = self.data_loader.load_data(train=False)

            self.loss_history = []
            self.train_loss_history = []
            self.time_steps = []

            for (x_train, y_train), (x_val, y_val) in zip(training_loader, validation_loader):

                # ===== Validation =====

                # forward and loss

                yhat_val = self.model.forward(x_val)

                val_error = MeanSquaredError().loss(y_true=y_val, y_pred=yhat_val)


                # ==== Training =====

                # forward and back 


                yhat_train = self.model.forward(x_train)
                dL_dyhat = self.loss_function.derivative(y_train, yhat_train)

                w_grads, b_grads = self.model.backward(loss_grad=dL_dyhat, input_data=x_train)

                # Update

                for i, (w_grad, b_grad) in enumerate(zip(w_grads, b_grads)):
                    self.model.layers[i].W -= self.learning_rate * w_grad
                    self.model.layers[i].b -= self.learning_rate * b_grad

                # Compute training loss for plotting
                train_loss = MeanSquaredError().loss(y_train, yhat_train)

                # # ===== Plotting =====
                #
                # # Update time and loss histories
                # current_time = self.time_steps[-1] + 1 if len(self.time_steps) > 0 else 0
                # self.time_steps.append(current_time)
                # self.loss_history.append(val_error.item())
                # self.train_loss_history.append(train_loss)

                # # Update the plot with both validation and training loss
                # ax = self.plot.gca()
                # ax.clear()
                # ax.set_title("Loss vs Time")
                # ax.set_xlabel("Time")
                # ax.set_ylabel("Loss")
                # ax.plot(self.time_steps, self.loss_history, marker='o', label="Validation Loss")
                # ax.plot(self.time_steps, self.train_loss_history, marker='x', color='r', label="Training Loss")
                # ax.legend()
                # plt.pause(0.01)

        # Return the hopefully  trained model
        return self.model



# ===== Test with simple, linear example first ===== 

from src.Functions.losses import SquaredError
from src.Functions import MeanSquaredError

# Synthetic Data

w_true = np.array([-1, 2, -5, 3], dtype=np.float32)
b_true = np.array([3], dtype=np.float32)

generator = SyntheticLinearDataGenerator(w_true, b_true, num_train=10, num_validate=20, batch_size=3)

# Model

h1 = Layer(4, 1, Linear())
# h2 = Layer(2, 2, Linear())
# h3 = Layer(2, 3, Linear())
neural_network = MultilayerPerceptron(layers=(h1,))

# Train

trainer = Train(model=neural_network, loss_function=SquaredError(), learning_rate=0.05, data_loader=generator)
trained_neural_network = trainer.train(epochs=2)

# Hold the plot open until you close it manually
# plt.show()