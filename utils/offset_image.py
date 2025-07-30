# Function to create a median offset of an image

import numpy as np


def offset_image(image, type="none"):
    # Make sure we are caps proof
    type = type.lower()

    # Calculate either the median or mean of the image, neglacting black / 0 pixels (cropped sides of organoids)
    if type == "median":
        value = np.median(image[image > 0])
    if type == "mean":
        value = np.mean(image[image > 0])

    # Offset the image by subtracting the value from all pixels
    offset = image.astype(np.float32) - value

    # Values below the cutoff value should be 0
    offset[offset < 0] = 0

    # Convert image back to 16 bit image
    offset = np.clip(offset, 0, 255).astype(np.uint16)

    return offset
