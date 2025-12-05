def data_2D_chunking(image):
    from skimage.measure import regionprops
    import pandas as pd

    data2d = []
    empty_slice_counter = 0

    for z in range(image.shape[0]):
        props = regionprops(image[z])

        if len(props) == 0:
            # Empty slice - add empty DataFrame with correct columns
            empty_df = pd.DataFrame(
                columns=[
                    "label",
                    "x",
                    "y",
                    "z",
                    "area",
                    "orientation",
                    "assignmentTag",
                    "majorAxisLength",
                    "minorAxisLength",
                    "indexinslice",
                    "z_index_without_empty",
                ]
            )
            data2d.append(empty_df)
            empty_slice_counter += 1
            continue

        slice_data = []
        for idx, prop in enumerate(props):
            slice_data.append(
                {
                    "label": prop.label,
                    "x": int(prop.centroid[1]),
                    "y": int(prop.centroid[0]),
                    "z": z,  # This is indexInZ in Julia
                    "area": prop.area,
                    "orientation": prop.orientation,
                    "assignmentTag": 0,
                    "majorAxisLength": prop.major_axis_length,
                    "minorAxisLength": prop.minor_axis_length,
                    "indexinslice": idx,  # This is indexData in Julia
                    "z_index_without_empty": z - empty_slice_counter,
                }
            )
        data2d.append(pd.DataFrame(slice_data))

    return data2d
