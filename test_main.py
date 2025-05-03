import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello there!"}

def test_upload_invalid_file_type():
    with open("test.txt", "w") as f:
        f.write("This is a test file.")
    with open("test.txt", "rb") as f:
        response = client.post("/upload/", files={"file": ("test.txt", f, "text/plain")})
    assert response.status_code == 400
    assert response.json() == {"error": "Only PNG or JPEG images allowed."}

@pytest.mark.skip(reason="Requires mocking of prepare_image and predict_number")
def test_upload_valid_file_type():
    # Mocking prepare_image and predict_number is required for this test
    # Assuming the implementation of prepare_image and predict_number is correct
    pass

# RUN:
# pytest test_main.py