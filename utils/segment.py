# File that goes over image slice by slice and applies cellpose

from alive_progress import alive_bar
import numpy as np


def segment(image, model):
    depthIm, heightIm, widthIm = image.shape

    masks_list = []

    with alive_bar(depthIm, title="Segmenting slices") as bar:
        for z in range(depthIm):
            plane_2d = image[z, :, :]
            mask, _, _ = model.eval(plane_2d, diameter=None, do_3D=False)
            masks_list.append(mask)
            bar()

    segmented_stack = np.stack(masks_list, axis=0).astype(np.uint16)

    return segmented_stack
