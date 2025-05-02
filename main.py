# Basic FastAPI Example to Receive an Uploaded Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from torchvision import transforms
import uvicorn
# from interface import NeuralNetwork

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello there!"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):

    # Check content type
    if file.content_type not in ["image/png", "image/jpeg"]:
        return JSONResponse(status_code = 400, content={"error": "Only PNG or JPEG images allowed."})
    
    # Read the file into a PIL image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L") 
    image = image.resize((28, 28))

    # Convert to NumPy or tensor as needed for model input
    # ...

    return {"filename": file.filename, "size": image.size}

# TO RUN:

# uvicorn main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port = 8000, reload = True)