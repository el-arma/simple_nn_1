# %%
# GET DATA

from torchvision import datasets


# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True
)

test_dataset  = datasets.MNIST(
    root="data",
    train=False,
    download=True
)


# %%
# Save imgs 

import os

# destination folder
output_dir = "mnist_images"

os.makedirs(output_dir, exist_ok=True)

# save first 100 imgages:
for i in range(100):
    img = test_dataset[i][0]
    label = test_dataset[i][1]

    # save image with lbl in name
    img.save(os.path.join(output_dir, f"{i:04d}_label_{label}.png"))


# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms



# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform to normalize and convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),                
    # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  
    # Normalize pixel values to [-1, 1]
])


# Load the MNIST training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True,
                                           transform=transform, download=True)

test_dataset = datasets.MNIST(root='./data', train=False,
                                          transform=transform, download=True)


# Data loaders for training and testing:
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = 64, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 1000, shuffle = False)

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
    
# Create the model instance and pass it to the device (CPU or GPU)
model = NeuralNetwork().to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)


total_train_loss = []

# Training loop:

def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()
    # Set model to training mode

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)


        loss.backward()
        # Backpropagation

        optimizer.step()
        # Update weights

        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            total_train_loss.append(loss)


    return None

def test(dataloader, model, loss_fn):

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches

    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return None

epochs = 20

for t in range(epochs):
    print(f"Epoch {t + 1}\n", "-" * 36)
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)


torch.save(model.state_dict(), "nn_model-v1.00.pth")

print("Done!")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(total_train_loss, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


# %%
import torch

torch.save(model.state_dict(), "nn_model-v1.00.pth")

print("Saved PyTorch Model State to nn_model-v1.00.pth")
