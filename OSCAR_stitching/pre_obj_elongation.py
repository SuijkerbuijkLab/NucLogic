def pre_obj_elongation(slice_idx, obj_idx, data2d, max_z, min_z, med_z):
    import pandas as pd
    import numpy as np
    from OSCAR_stitching.ellipses_connector import ellipses_connector
    from OSCAR_stitching.info_preobj_dist_z import info_preobj_dist_z

    # Create deep copy to avoid modifying original data2d
    data2d = [df.copy() for df in data2d]

    slice_data = data2d[slice_idx]

    volumes = []
    pearson_ind = 0
    volume_elipse = (
        slice_data.iloc[obj_idx]["majorAxisLength"]
        * slice_data.iloc[obj_idx]["minorAxisLength"]
        * np.pi
    )
    temp_elongated = slice_data.iloc[
        [obj_idx]
    ].copy()  # starts out as just the elipse in 1 slice

    # Store the starting slice index
    starting_slice = slice_idx
    current_slice = slice_idx  # Track current position

    # Loop until max_z slices or end of data2d
    # FIX: Check against starting_slice + max_z and len(data2d) - 1
    while (
        current_slice < starting_slice + max_z - 1 and current_slice < len(data2d) - 1
    ):
        ellipse_info = ellipses_connector(
            data2d, current_slice, temp_elongated, obj_idx, min_z
        )
        continue_elongation = ellipse_info["continue_elongation"]
        next_ellipse_idx = 0  # Initialize

        if continue_elongation == 1:
            # Connection found - extend the elongated object
            next_ellipse_idx = ellipse_info["next_ellipse_idx"]

            # Update for next iteration
            obj_idx = next_ellipse_idx  # Update obj_idx for next connector call
            current_slice += 1  # Move to next slice

            # Append connected ellipse to elongated object
            new_row = data2d[current_slice].iloc[[obj_idx]].copy()
            temp_elongated = pd.concat([temp_elongated, new_row], ignore_index=True)

            # Calculate cumulative volume (sum of all ellipse areas)
            cumulative_volume = (
                temp_elongated["majorAxisLength"]
                * temp_elongated["minorAxisLength"]
                * np.pi
            ).sum()
            volumes.append(cumulative_volume)
        else:
            # No connection found - stop elongation
            break

    info = info_preobj_dist_z(temp_elongated, index1=0, index2=int(round(med_z)))
    distances_from_line = info["dist"]
    angles_between_segments = info["ang"]
    pearson = info["pearson"]

    # Return everything
    return {
        "temp_elongated": temp_elongated,
        "volumes": volumes,
        "distances_from_line": distances_from_line,
        "angles_between_segments": angles_between_segments,
        "pearson": pearson,
    }
