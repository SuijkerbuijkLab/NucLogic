import os, argparse
import numpy as np
from cellpose import io, transforms
import tifffile


def make_train_3d(
    input_image,
    input_mask,
    nimg_per_tif=10,
    crop_size=512,
    channel_axis=-1,
    z_axis=0,
    anisotropy=1.0,
    tile_norm=0,
    sharpen_radius=0.0,
):
    np.random.seed(0)
    nimg_per_tif = nimg_per_tif
    crop_size = crop_size
    dirname = os.path.split(input_image)[0]
    os.makedirs(os.path.join(dirname, "train/"), exist_ok=True)
    pm = [(0, 1, 2, 3), (2, 0, 1, 3), (1, 0, 2, 3)]
    npm = ["YX", "ZY", "ZX"]
    name = os.path.splitext(os.path.split(input_image)[-1])[0]
    img0 = io.imread_3D(input_image)
    mask0 = io.imread_3D(input_mask)

    try:
        img0 = transforms.convert_image(
            img0, channel_axis=channel_axis, z_axis=z_axis, do_3D=True
        )
        mask0 = transforms.convert_image(
            mask0, channel_axis=channel_axis, z_axis=z_axis, do_3D=True
        )
    except ValueError:
        print(
            "Error converting image. Did you provide the correct --channel_axis and --z_axis ?"
        )

    for p in range(3):
        img = img0.transpose(pm[p]).copy()
        mask = mask0.transpose(pm[p]).copy()
        print(npm[p], img[0].shape)
        Ly, Lx = img.shape[1:3]
        indices = np.random.permutation(img.shape[0])[:nimg_per_tif]
        imgs = img[indices]
        masks = mask[indices]
        if anisotropy > 1.0 and p > 0:
            imgs = transforms.resize_image(imgs, Ly=int(anisotropy * Ly), Lx=Lx)
            # Use nearest neighbor for masks to preserve integer labels
            from scipy.ndimage import zoom

            zoom_factors = (
                (1, anisotropy, 1) if len(masks.shape) == 3 else (1, anisotropy, 1, 1)
            )
            masks = zoom(masks, zoom_factors, order=0)
        for k, (img, mask) in enumerate(zip(imgs, masks)):
            if tile_norm:
                img = transforms.normalize99_tile(img, blocksize=tile_norm)
            if sharpen_radius:
                img = transforms.smooth_sharpen_img(img, sharpen_radius=sharpen_radius)
            ly = 0 if Ly - crop_size <= 0 else np.random.randint(0, Ly - crop_size)
            lx = 0 if Lx - crop_size <= 0 else np.random.randint(0, Lx - crop_size)
            io.imsave(
                os.path.join(dirname, f"train/{name}_{npm[p]}_{k}.tif"),
                img[ly : ly + crop_size, lx : lx + crop_size].squeeze(),
            )
            io.imsave(
                os.path.join(dirname, f"train/{name}_{npm[p]}_{k}_masks.tif"),
                mask[ly : ly + crop_size, lx : lx + crop_size].squeeze(),
            )


make_train_3d(
    input_image=r"C:\Users\6331823\Local SSD\Data_Merel\EXP087\20250604_Lyz_AldoB_DAPI.40x_F26\cropped_nuclei.tif",
    input_mask=r"C:\Users\6331823\Local SSD\Data_Merel\EXP087\20250604_Lyz_AldoB_DAPI.40x_F26\cropped_nuclei_cp_masks.tif",
    anisotropy=8.125,
)
