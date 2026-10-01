from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from app.predict import Predictor


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "ADAM_model.pth"

predictor = None


# Load model when application starts
@asynccontextmanager
async def lifespan(app: FastAPI):

    global predictor

    predictor = Predictor(MODEL_PATH)

    yield

    predictor = None


# Create FastAPI application
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="CNN image classification using PyTorch",
    version="1.0.0",
    lifespan=lifespan
)


# Health check endpoint
@app.get("/")
def home():

    return {
        "message": "Cats vs Dogs API is running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


# Prediction endpoint
@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    # Validate file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=415,
            detail="Only JPG, PNG and WEBP images are supported"
        )

    # Read uploaded image
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image file"
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB"
        )

    try:
        # Call prediction service
        result = predictor.predict(image_bytes)

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )