from torch import cuda, device, load, no_grad
from model.NNmodel import NeuralNetwork
from torchvision import transforms
from PIL import Image

model = NeuralNetwork().to(device('cuda' if cuda.is_available() else 'cpu'))
model.load_state_dict(load("model/nn_model-v1.00.pth", weights_only=True))

# Transform to normalize and convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),                
    # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  
    # Normalize pixel values to [-1, 1]
])

# Function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert('L')  
    # Convert to grayscale if needed
    image = transform(image).unsqueeze(0)       
    # Apply transformations and add batch dimension
    return image.to(device('cuda' if cuda.is_available() else 'cpu'))

# Example usage
image_path = "data/mnist_images/0000_label_7.png"
input_image = load_image(image_path)

model.eval()

with no_grad():
    prediction = model(input_image)
    predicted_class = prediction.argmax(1).item()
    print(f"Predicted class: {predicted_class}")