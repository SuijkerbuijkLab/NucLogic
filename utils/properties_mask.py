# Function to get properties from masked images and put them in a pd df

from skimage.measure import regionprops
import pandas as pd


def properties_mask(image):
    props = regionprops(image)

    data = []
    for prop in props:
        label = prop.label
        centroid = prop.centroid  # (z, y, x)
        bounding_box = prop.bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)
        volume = prop.area

        data.append(
            {
                "label": label,
                "z": centroid[0],
                "y": centroid[1],
                "x": centroid[2],
                "bounding_box": bounding_box,
                "volume": volume,
            }
        )

    df_props = pd.DataFrame(data)

    return df_props
