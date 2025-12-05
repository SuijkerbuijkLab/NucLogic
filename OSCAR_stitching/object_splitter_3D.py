from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class Objects3D:
    """
    Container for validated 3D objects and metadata.

    Attributes:
        preok: List of DataFrames, each representing a validated 3D object
        infoDF: DataFrame with summary statistics (original length, cut point, coherence)
        inis: Total number of initiator objects processed
        noInits: Array of merged object coordinates
    """

    preok: list  # List of pd.DataFrame
    infoDF: pd.DataFrame
    inis: int
    noInits: np.ndarray


def object_splitter_3D(image, min_z=4, med_z=6, max_z=9, nuclei_channel=0):
    from tqdm import tqdm
    import numpy as np
    from OSCAR_stitching.data_2D_chunking import data_2D_chunking
    from OSCAR_stitching.pre_obj_elongation import pre_obj_elongation
    from OSCAR_stitching.check_intit import check_intit
    from OSCAR_stitching.terminator_returns import terminator_returns
    from OSCAR_stitching.get_covariance import get_covariance
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data2d = data_2D_chunking(image)

    # Storage containers for final 3D objects and their metrics
    validated_objects_3d = []  # List of DataFrames, each representing a 3D object
    cut_points_f = []  # List of integers: where each object was truncated
    coherence = []  # List of floats: Pearson correlation for each object
    object_lengths = []  # List of integers: number of slices per object

    # Track which objects have been assigned
    # assigned = [np.zeros(len(props), dtype=bool) for props in data2d.groupby("z")]

    # Global counters
    border_effect_low = 1
    initiators = 0
    merged_objects = np.zeros((1, 3))  # Objects that merged/disappeared
    counter_cin = 0
    count_pearson = 0
    counter_truncations = 0

    print("Starting 3D object stitching...")
    # Main loop to process each slice
    for slice_idx in tqdm(range(len(data2d) - border_effect_low)):
        current_slice = data2d[slice_idx]  # Get DataFrame for this slice
        num_objects = len(current_slice)  # Number of rows in this DataFrame

        for obj_idx in range(num_objects):  # Loop through each object in the slice

            # This function checks if the object merges or disappears in the slice below
            disappear_init = check_intit(data2d, slice_idx, obj_idx)
            # If merger detected, record it
            if np.any(disappear_init != 0):
                merged_objects = np.vstack([merged_objects, disappear_init])

            # Go over any object that is not stitched yet
            if current_slice.iloc[obj_idx]["assignmentTag"] == 0:
                initiators += 1

                # Perform 3D elongation
                pre_obj = pre_obj_elongation(
                    slice_idx, obj_idx, data2d, max_z, min_z, med_z
                )

                temp_elongated = pre_obj["temp_elongated"]  # Elongated object data2d
                pre_vol = pre_obj["volumes"]  # Volume progression
                dist_v = pre_obj["distances_from_line"]  # Distances from fitted line
                ang_v = pre_obj["angles_between_segments"]  # Angles between segments
                pearson = pre_obj["pearson"]  # Coherence score

                # Initialize truncation point (full length by default)
                cut_point = len(temp_elongated)
                factor = pearson

                # Check if the elongated object meets minimum length
                if len(temp_elongated) >= min_z:
                    # If longer than median length, evaluate for possible truncation
                    if len(temp_elongated) > med_z:
                        # Detect potential merge points
                        terminator_info = terminator_returns(
                            dist_v[1:],  # Skip first distance
                            ang_v,
                            int(round(med_z)),
                        )

                        cin = terminator_info["numbIn"]
                        cout = terminator_info["numbOut"]

                        # Normalize by segment length
                        cin = cin / med_z
                        cout = cout / (len(temp_elongated) - med_z + 1)

                        cut_positions = terminator_info["cutOut"]
                        outlier_lengths = terminator_info["lengthOut"]

                        # Check if later part is more erratic than beginning
                        if cin < cout and cout > 0:
                            counter_cin += 1

                            # Get first outlier position
                            n_point_h = int(cut_positions[0] + 1)

                            # Calculate Pearson BEFORE cut point
                            if n_point_h - 2 < 0:
                                pearson_in = 0
                            else:
                                pearson_in = get_covariance(
                                    dist_v[1 : n_point_h - 1],
                                    ang_v[0 : n_point_h - 2],
                                )

                            # Calculate Pearson AFTER cut point
                            if n_point_h - 1 < 0:
                                pearson_out = 0
                            else:
                                pearson_out = get_covariance(
                                    dist_v[n_point_h - 1 :],
                                    ang_v[n_point_h - 2 :],
                                )

                            # Truncate if trajectory before cut is better
                            if pearson_in > pearson:
                                count_pearson += 1
                                cut_point = int(n_point_h)
                                counter_truncations += 1

                    # ===== STORE OBJECT METADATA =====
                    cut_points_f.append(cut_point)
                    object_lengths.append(len(temp_elongated))
                    coherence.append(pearson)

                    # ===== MARK ELLIPSES AS ASSIGNED =====
                    for zz in range(cut_point):
                        # Use the actual column names from temp_elongated
                        slice_z = int(temp_elongated.iloc[zz]["z"])  # This is indexInZ
                        data_idx = int(
                            temp_elongated.iloc[zz]["indexinslice"]
                        )  # This is indexData

                        # Increment assignment tag in original data
                        data2d[slice_z].at[data_idx, "assignmentTag"] += 1

                        # Update tag in elongated object
                        temp_elongated.at[zz, "assignmentTag"] = data2d[slice_z].iloc[
                            data_idx
                        ]["assignmentTag"]

                    # ===== STORE FINAL OBJECT (truncated if necessary) =====
                    final_object = temp_elongated.iloc[:cut_point].copy()
                    validated_objects_3d.append(final_object)
    # Create summary DataFrame
    summary_df = pd.DataFrame(
        {
            "original_length": object_lengths,
            "cut_point": cut_points_f,
            "pearson": coherence,
        }
    )

    print(f"\n=== Summary ===")
    print(f"Initiators found: {initiators}")
    print(f"Valid 3D objects: {len(validated_objects_3d)}")
    print(f"Objects truncated: {counter_truncations}")
    print(f"Cin/Cout triggers: {counter_cin}")
    print(f"Pearson-based truncations: {count_pearson}")

    # ===== RETURN OBJECTS3D DATACLASS =====
    return (
        Objects3D(
            preok=validated_objects_3d,
            infoDF=summary_df,
            inis=initiators,
            noInits=merged_objects,
        ),
        counter_cin,
        count_pearson,
    )
