# Function to get properties from masked images and put them in a pd df

from skimage.measure import regionprops_table
import pandas as pd
import tifffile


def properties_channel(mask, image, type=None):
    image = tifffile.imread(image)

    props_channel = regionprops_table(
        mask, intensity_image=image, properties=["label", "area", "mean_intensity"]
    )

    props_channel = pd.DataFrame(props_channel).rename(
        columns={"area": f"area_{type}", "mean_intensity": f"raw_{type}"}
    )

    return props_channel
