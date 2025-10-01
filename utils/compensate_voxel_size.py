# Function to change a properties dataframe to compensate for voxel size in Z, Y, and X


def compensate_voxel_size(df, voxel_size):
    z_size, y_size, x_size = voxel_size

    # Compensate for voxel size in the volume
    df["volume"] = df["volume"] * z_size * y_size * x_size

    # Compensate for voxel size in the coordinates
    df["z"] = df["z"] * z_size
    df["y"] = df["y"] * y_size
    df["x"] = df["x"] * x_size

    return df
