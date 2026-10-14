# File that goes over an image slice by slice and applies cellpose

import numpy as np
import warnings
import logging

logging.getLogger("cellpose").setLevel(logging.ERROR)


def segment(image, model):
    # Get the shape of the image to find amount of Z slices
    depthIm = image.shape[0]

    # Keep track of mask of every z slice
    masks_list = []

    # Loop over every Z slice and run the cellpose model on that
    for z in range(depthIm):
        plane_2d = image[z, :, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask, _, _ = model.eval(
                plane_2d,
                diameter=None,
                normalize=True,
                flow_threshold=0.4,
                # invert=False,
                resample=True,
                do_3D=False,
                progress=None,
            )

        # Add mask to the mask list
        masks_list.append(mask)

    # Stack the mask list into an actual 3D image again
    segmented_stack = np.stack(masks_list, axis=0).astype(np.uint16)

    return segmented_stack


def segment_3d(image, model, anisotropy=1.0, flow3d_smooth=0.4):
    """Cellpose's own 3D mode: one call on the whole (Z, Y, X) stack, no NucLogic stitching.

    anisotropy = Z voxel size / XY voxel size, so Cellpose can rescale Z to match XY.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask, _, _ = model.eval(
            image,
            z_axis=0,
            do_3D=True,
            anisotropy=anisotropy,
            flow3D_smooth=flow3d_smooth,
            diameter=None,
            normalize=True,
            progress=None,
        )
    return mask.astype(np.uint32 if mask.max() > np.iinfo(np.uint16).max else np.uint16)
