# Function to load cellpose models
import os

# Sentinel used instead of a file path when the built-in Cellpose-SAM
# foundation model is selected (it ships with the environment, so there is
# nothing in the models folder to point at).
CELLPOSE_SAM_MODEL_NAME = "Cellpose-SAM foundation model"


def get_device():
    """Torch device for Cellpose and SAM 2: NVIDIA CUDA, then Apple-silicon MPS, else CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path=None):
    import torch
    from cellpose import models

    device = get_device()
    print(f"Using {device.upper()} for processing.")

    if model_path and model_path != CELLPOSE_SAM_MODEL_NAME:
        model = models.CellposeModel(
            gpu=(device != "cpu"), pretrained_model=model_path, device=torch.device(device)
        )
        print(f"loaded custom model: {os.path.basename(model_path)}")
    else:
        model = models.CellposeModel(gpu=(device != "cpu"), device=torch.device(device))
        print(f"loaded {CELLPOSE_SAM_MODEL_NAME}")
    return model
