from torch import cuda, device, load, no_grad
from model.NNmodel import NeuralNetwork
from torchvision import transforms
from PIL import Image
import io

server_device = device('cuda' if cuda.is_available() else 'cpu')

model = NeuralNetwork().to(server_device)

model.load_state_dict(load("model/nn_model-v1.00.pth", weights_only=True))

# Transform to normalize and convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),                
    # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  
    # Normalize pixel values to [-1, 1]
])

# Function to load and preprocess an image
def prepare_image(image_content):

    image = Image.open(io.BytesIO(image_content)).convert("L") 
    # Convert to grayscale if needed

    image = transform(image).unsqueeze(0)       
    # Apply transformations and add batch dimension

    return image.to(server_device)


def predict_number(input_image):

    model.eval()

    with no_grad():
        prediction = model(input_image)

        predicted_number = prediction.argmax(1).item()

        return f'Recognised number: {predicted_number}'

