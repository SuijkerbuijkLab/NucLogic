# Function to change a properties dataframe to compensate for voxel size in Z, Y, and X


def compensate_voxel_size(df, voxel_size):
    z_size, y_size, x_size = voxel_size

    # Compensate for voxel size in the volume
    df["volume"] = df["volume"] * z_size * y_size * x_size

    # Change the name of ZYX columns to pixel columns
    df.rename(columns={"z": "z_pixel", "y": "y_pixel", "x": "x_pixel"}, inplace=True)

    # Compensate for voxel size in the coordinates
    df["z"] = df["z_pixel"] * z_size
    df["y"] = df["y_pixel"] * y_size
    df["x"] = df["x_pixel"] * x_size

    return df
