# 🐱 Cats vs Dogs Image Classification API

A deep learning image classification project built with **PyTorch** and **FastAPI** to classify images into two categories: **Cats** and **Dogs**.

The project uses a custom Convolutional Neural Network (CNN) trained with the **Adam optimizer** and provides a REST API for real-time image inference.

## 🚀 Project Overview

This project demonstrates how to deploy a trained deep learning model using FastAPI with a modular and maintainable project structure.

**Key Features**

* Custom CNN architecture implemented using PyTorch.
* Binary image classification (Cats vs Dogs).
* Model inference using pretrained weights saved after Adam optimization.
* Image preprocessing and tensor conversion.
* REST API built with FastAPI.
* Input validation and error handling.
* Interactive API documentation using Swagger UI.
* Modular architecture separating the model, preprocessing, prediction, and API layers.

## 🏗️ Project Architecture

```text
Cats_Dogs_Classification/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   └── preprocessing.py
│
├── models/
│   └── ADAM_model.pth
│
│
└── README.md
```

### 📂 Components

| File               | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| `main.py`          | Defines FastAPI application and API endpoints.                          |
| `model.py`         | Defines the custom CNN architecture.                                    |
| `preprocessing.py` | Handles image decoding, resizing, normalization, and tensor conversion. |
| `predict.py`       | Loads the trained model and performs inference.                         |
| `ADAM_model.pth`   | Stores the trained model weights.                                       |
## 🧠 Model Architecture

The project uses a custom CNN architecture consisting of:

* **4 Convolutional Blocks**

  * Conv2D
  * Batch Normalization
  * ReLU Activation
  * Max Pooling

* **Global Average Pooling**

  * Adaptive Average Pooling
  * Flatten

* **Fully Connected Layers**

  * Linear (256 → 128)
  * Batch Normalization
  * ReLU
  * Linear (128 → 64)
  * Batch Normalization
  * ReLU
  * Dropout
  * Linear (64 → 1)

**Training Configuration**

* Framework: PyTorch
* Optimizer: Adam
* Loss Function: BCEWithLogitsLoss
* Output: Single logit for binary classification
* Inference Activation: Sigmoid

## ⚙️ Image Preprocessing

Uploaded images pass through the following preprocessing pipeline:

1. Decode the uploaded image.
2. Convert the image to grayscale.
3. Resize the image to **375 × 448** pixels.
4. Normalize pixel values to the range **[0, 1]**.
5. Convert the image into a PyTorch tensor.
6. Add the batch dimension before model inference.

## 🔌 API Endpoints

| Method | Endpoint   | Description                                                      |
| ------ | ---------- | ---------------------------------------------------------------- |
| GET    | `/`        | Returns a welcome message.                                       |
| GET    | `/health`  | Checks API and model status.                                     |
| POST   | `/predict` | Accepts an image and returns the predicted class and confidence. |

### Example Prediction Response

```json
{
  "predicted_class": "dogs",
  "confidence": 0.92,
  "probabilities": {
    "cats": 0.08,
    "dogs": 0.92
  }
}
```

*Example response for illustration only.*

## 🛠️ Technologies Used

* **Python** — Core programming language
* **PyTorch** — Deep learning model development and inference
* **FastAPI** — REST API development
* **OpenCV** — Image processing
* **NumPy** — Numerical operations
* **Uvicorn** — ASGI server


## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Cats_Dogs_Classification.git

cd Cats_Dogs_Classification
```

### 2. Add the Trained Model

Place your trained model checkpoint in the following location:

```text
models/ADAM_model.pth
```

The checkpoint is required to run inference. It is not included in the repository by default.

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## 📖 API Documentation

FastAPI automatically generates interactive API documentation.

After starting the server, open:

**Swagger UI:**

```text
http://127.0.0.1:8000/docs
```

You can use the interface to upload images and test the prediction endpoint directly.

## 🔮 Future Improvements

* Add a Streamlit frontend for interactive image classification.
* Containerize the application using Docker.
* Add model performance evaluation and confidence calibration.
* Implement logging and monitoring.
* Deploy the API to a cloud platform.
* Add continuous integration using GitHub Actions.

## 👨‍💻 Author

**Mohamed Abd Elrhman**

Computer and Systems Engineering Graduate | Data Science & Machine Learning

[GitHub](https://github.com/Mohamed-AbdElrhman49)

---

⭐ If you find this project useful, feel free to explore the repository and share your feedback.
