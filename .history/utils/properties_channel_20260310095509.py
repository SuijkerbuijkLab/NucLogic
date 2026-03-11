# Function to get properties from masked images and corresponding channels and put them in a pd df

import pandas as pd
from skimage.measure import regionprops_table


def properties_channel(mask, image, type=None):

    # Get the properties of the mask for the channel you specified (image)
    props_channel = regionprops_table(
        mask, intensity_image=image, properties=["label", "mean_intensity"]
    )

    # Make this into a pandas dataframe and rename some variables to the name of the channel for later merging
    props_channel = pd.DataFrame(props_channel).rename(
        columns={
            "label": "label",
            "mean_intensity": f"{type}_mean_intensity",
        }
    )

    return props_channel
