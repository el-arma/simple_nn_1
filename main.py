# Basic FastAPI Example to Receive an Uploaded Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from interface import prepare_image, predict_number

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
    image_content = await file.read()

    transformed_img = prepare_image(image_content)

    predicted_result = predict_number(transformed_img)

    return predicted_result

# TO RUN MANUALLY:
# uvicorn main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port = 8000, reload = True)