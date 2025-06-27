# Function to do a mean threshold on an image and apply this

import numpy as np


def threshold(image):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    threshold_value = mean_intensity + std_intensity
    image[image < threshold_value] = 0

    return image
