import cv2
import numpy as np
import torch


def preprocess_image(image_bytes):

    # Convert bytes to numpy array
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # Read image as grayscale
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError("Invalid image file")

    # Resize
    image = cv2.resize(
        image,
        (448, 375)
    )

    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert to tensor: (C, H, W)
    image = torch.from_numpy(image).unsqueeze(0)

    # Add batch dimension: (N, C, H, W)
    image = image.unsqueeze(0)

    return image