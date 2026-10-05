# Function to load cellpose models
import os

# Sentinel used instead of a file path when the built-in Cellpose-SAM
# foundation model is selected (it ships with the environment, so there is
# nothing in the models folder to point at).
CELLPOSE_SAM_MODEL_NAME = "Cellpose-SAM foundation model"


def load_model(model_path=None):
    import torch
    from cellpose import models

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        gpu = True
    elif torch.mps.is_available():
        device = "mps"
        gpu = True
    else:
        gpu = False

    print(f"Using {device.upper()} for processing.")

    if model_path and model_path != CELLPOSE_SAM_MODEL_NAME:
        model = models.CellposeModel(
            gpu=gpu, pretrained_model=model_path
        )
        print(f"loaded custom model: {os.path.basename(model_path)}")
    else:
        model = models.CellposeModel(gpu=gpu)
        print(f"loaded {CELLPOSE_SAM_MODEL_NAME}")
    return model
