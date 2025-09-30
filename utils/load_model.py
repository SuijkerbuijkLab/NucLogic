# Function to load cellpose models
import os

import torch
from cellpose import models


def load_model(model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()} for processing.")

    if model_path:
        model = models.CellposeModel(
            gpu=(device == "cuda"), pretrained_model=model_path
        )
        print(f"loaded custom model: {os.path.basename(model_path)}")
    else:
        model = models.CellposeModel(gpu=(device == "cuda"))
        print("loaded default SAM model")
    return model
