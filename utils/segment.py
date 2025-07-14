# File that goes over image slice by slice and applies cellpose

import numpy as np


def segment(image, model):
    depthIm, heightIm, widthIm = image.shape

    masks_list = []

    for z in range(depthIm):
        plane_2d = image[z, :, :]
        mask, _, _ = model.eval(plane_2d, diameter=None, do_3D=False)
        masks_list.append(mask)
        
    segmented_stack = np.stack(masks_list, axis=0).astype(np.uint16)

    return segmented_stack
