# Function to get properties from masked images and put them in a pandas df

import pandas as pd
from skimage.measure import regionprops


def properties_mask(image):

    # Get the properties of every mask in the image
    props = regionprops(image)

    data = []

    # For every mask found, get the label, centeroid, boundingbox, and volume
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

    # Make a pandas data frame from the data
    columns = ["label", "z", "y", "x", "bounding_box", "volume"]
    df_props = pd.DataFrame(data, columns=columns)

    return df_props
