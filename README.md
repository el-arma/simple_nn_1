# AGH-post_grad-MSI2

**[MSI2] Simple Neural Network Prediction Model + FastAPI**  
Project developed as part of postgraduate studies at AGH University of Science and Technology.

## 📌 Overview

This repository contains a minimal implementation of a neural network-based prediction model using [PyTorch](https://pytorch.org/) and [FastAPI](https://fastapi.tiangolo.com/), tested on the legendary MNIST dataset.

## 🧠 Project Components

- **Neural Network Model**: A simple feed-forward neural network for number prediction.
- **FastAPI Backend**: RESTful API exposing the model's capabilities.

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/el-arma/simple_nn_1
   cd simple_nn_1

2. **Prep evn. (if you use uv)**
   ```bash 
   uv sync

2. **Get data**
   ```bash 
   cd data
   python -m get_data
   ```

3. **Run the server**
   ```bash 
   python -m main
   ```

4. **Run the client**
   ```bash 
   python -m client_side
   ```