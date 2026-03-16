# Function to create a median offset of an image

import numpy as np


def offset_image(image, type="none"):
    # Make sure we are caps proof
    type = type.lower()

    positive_pixels = image[image > 0]
    if positive_pixels.size == 0:
        return np.zeros_like(image)

    # Calculate either the median or mean of the image, neglacting black / 0 pixels (cropped sides of organoids)
    if type == "median":
        value = np.median(positive_pixels)
    elif type == "mean":
        value = np.mean(positive_pixels)
    else:
        value = 0

    # Offset the image by subtracting the value from all pixels
    offset = image.astype(np.float32) - value

    # Values below the cutoff value should be 0
    offset[offset < 0] = 0

    # Convert image back to source integer dtype range (e.g., uint16 up to 65535)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        offset = np.clip(offset, 0, max_value).astype(image.dtype)

    return offset
