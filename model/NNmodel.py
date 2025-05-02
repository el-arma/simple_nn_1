import torch.nn as nn

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #  defines a stack of layers applied one after another.
            nn.Linear(28 * 28, 128),
            # first layer that maps the 784 input features (from a 28x28 image) to 128 hidden units.
            nn.ReLU(),
            nn.Linear(128, 64),
            # a second hidden layer
            nn.ReLU(),
            nn.Linear(64, 10)
            # output layer, mapping to 10 output classes (e.g., for digit classification 0–9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits