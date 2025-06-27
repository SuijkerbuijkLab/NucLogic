# Function to load cellpose models
from cellpose import models
import torch


def load_model(custom_model=False, model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()} for processing.")

    if custom_model and model_path:
        model = models.CellposeModel(
            gpu=(device == "cuda"), pretrained_model=model_path
        )
    else:
        model = models.CellposeModel(gpu=(device == "cuda"))
    return model
