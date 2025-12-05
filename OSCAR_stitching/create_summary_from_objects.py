def create_summary_from_objects(objects_3d):
    """
    Creates a summary DataFrame with geometric properties for each 3D object.

    Args:
        objects_3d: Objects3D dataclass containing validated 3D objects

    Returns:
        DataFrame with columns needed for visualization:
        - Xcenter, Ycenter, Zcenter: Center coordinates
        - A, B: Major and minor axis lengths
        - Vz: Object length in Z
        - AngleXY: Orientation angle in XY plane
        - Vx, Vy: Direction vector components
        - C: Magnitude of direction vector
        - Label: Object ID
    """
    import numpy as np
    import pandas as pd
    from OSCAR_stitching.fit_3D_line import fit_3D_line

    summary_data = []

    for obj_id, obj_df in enumerate(objects_3d.preok, start=1):
        if len(obj_df) == 0:
            continue

        # Calculate center coordinates (mean position)
        x_center = obj_df["x"].mean()
        y_center = obj_df["y"].mean()
        z_center = obj_df["z"].mean()

        # Calculate mean axis lengths
        a = obj_df["majorAxisLength"].mean()
        b = obj_df["minorAxisLength"].mean()

        # Object length in Z
        v_z = len(obj_df)

        # regionprops orientation is angle from Y-axis (rows) in radians [-π/2, π/2]
        # Convert to angle from X-axis by adding π/2, then store in degrees
        angle_xy = np.degrees(obj_df["orientation"].mean() + np.pi / 2)

        # Calculate direction vector
        if len(obj_df) > 1:
            # Fit 3D line to get direction
            line = fit_3D_line(
                x=obj_df["x"].values, y=obj_df["y"].values, z=obj_df["z"].values
            )
            vx = line["vx"]
            vy = line["vy"]
            v_module = line["vmodule"]
        else:
            # Single slice object
            vx = 0.0
            vy = 0.0
            v_module = 1.0

        summary_data.append(
            {
                "Label": obj_id,
                "Xcenter": x_center,
                "Ycenter": y_center,
                "Zcenter": z_center,
                "A": a,
                "B": b,
                "Vz": v_z,
                "AngleXY": angle_xy,
                "Vx": vx,
                "Vy": vy,
                "C": v_module,
            }
        )

    return pd.DataFrame(summary_data)
