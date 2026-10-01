import torch

from pathlib import Path
from app.model import PyTorchModel
from app.preprocessing import preprocess_image


class Predictor:

    def __init__(self, model_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize architecture
        self.model = PyTorchModel().to(self.device)

        # Load Adam-trained weights
        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict)

        # Inference mode
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_bytes):

        # Preprocessing
        image = preprocess_image(image_bytes)

        image = image.to(self.device)

        # Forward pass
        output = self.model(image)

        # Convert logits to probability
        probability = torch.sigmoid(output).item()

        # IMPORTANT:
        # Confirm the class-to-index mapping from your notebook.
        if probability >= 0.5:
            predicted_class = "dogs"
        else:
            predicted_class = "cats"

        confidence = (
            probability
            if probability >= 0.5
            else 1 - probability
        )

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "cats": round(1 - probability, 4),
                "dogs": round(probability, 4)
            }
        }