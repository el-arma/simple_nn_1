import requests
import os

PORT = 8000

ENDPOINT = 'upload'

url = f'http://localhost:{PORT}/{ENDPOINT}'

def send_file(url, file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'image/png')}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")

    return response.status_code, response.text

mnist_images_path = "data/mnist_images"

# List all names in the folder
file_names = os.listdir(mnist_images_path)

for file_name in file_names:
    
    file_path = f'data/mnist_images/{file_name}'

    lbl = file_name[-5]

    send_file(url, file_path)

    print("-" * 35, '\n', "s/b >>> ", lbl)


