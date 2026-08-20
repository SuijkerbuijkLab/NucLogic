import pandas as pd
from skimage.measure import regionprops


def get_extra_mask_properties(
    mask, intensity_image=None, extra_props=None, channel_name=None, voxel_size=None
):

    # Get the properties of every mask in the image. Passing ``spacing`` makes
    # all spatial properties (lengths, volumes, positions, moments) come out in
    # physical units; dimensionless ratios, voxel counts, and intensities are
    # unaffected by it.
    props = regionprops(mask, intensity_image=intensity_image, spacing=voxel_size)
    extra_props = extra_props or []

    # Compatibility aliases for renamed properties across scikit-image versions.
    aliases = {
        "intensity_mean": ["intensity_mean", "mean_intensity"],
        "mean_intensity": ["mean_intensity", "intensity_mean"],
        "intensity_std": ["intensity_std"],
    }

    data = []

    # For every mask found, get the label, centeroid, boundingbox, and volume
    for prop in props:
        label = prop.label

        row = {
            "label": label,
        }

        for key in extra_props:
            if key in row:
                continue

            candidates = aliases.get(key, [key])
            value = None
            for candidate in candidates:
                if hasattr(prop, candidate):
                    value = getattr(prop, candidate)
                    break

            output_key = key
            if channel_name:
                output_key = f"{channel_name.lower()}_{key}"

            row[output_key] = value

        data.append(row)

    # Make a pandas data frame from the data
    if channel_name:
        columns = ["label", *[f"{channel_name.lower()}_{key}" for key in extra_props]]
    else:
        columns = ["label", *extra_props]
    df_props = pd.DataFrame(data, columns=columns)

    return df_props
