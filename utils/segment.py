# File that goes over an image slice by slice and applies cellpose

import numpy as np


def segment(image, model):
    # Get the shape of the image to find amount of Z slices
    depthIm, _, _ = image.shape

    # Keep track of mask of every z slice
    masks_list = []

    # Loop over every Z slice and run the cellpose model on that
    for z in range(depthIm):
        plane_2d = image[z, :, :]
        mask, _, _ = model.eval(
            plane_2d,
            diameter=None,
            normalize=True,
            flow_threshold=0.4,
            invert=False,
            resample=True,
            do_3D=False,
            progress=None,
        )

        # Add mask to the mask list
        masks_list.append(mask)

    # Stack the mask list into an actual 3D image again
    segmented_stack = np.stack(masks_list, axis=0).astype(np.uint16)

    return segmented_stack
