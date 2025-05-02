# %%
# GET DATA

from torchvision import datasets


# Download training data from open datasets.
training_data = datasets.MNIST(
    root=".",
    train=True,
    download=True
)

test_dataset  = datasets.MNIST(
    root=".",
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