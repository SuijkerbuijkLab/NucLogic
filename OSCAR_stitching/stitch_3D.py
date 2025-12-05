def stitch_3D(image, min_z=4, med_z=6, max_z=9, label_ellipses=True):
    """
    Convenience function to perform complete 3D object stitching.

    Args:
        image: 3D numpy array (Z, Y, X)
        min_z: Minimum z-slices for object validation
        med_z: Medium threshold for z-slices
        max_z: Maximum z-slices threshold
        label_ellipses: Whether to label each ellipse uniquely

    Returns:
        Labeled 3D image with stitched objects
    """
    from .object_splitter_3D import object_splitter_3D
    from .create_summary_from_objects import create_summary_from_objects
    from .draw_3D_ellipses import draw_3D_ellipses

    # Split 3D objects
    object3D, _, _ = object_splitter_3D(image, min_z=min_z, med_z=med_z, max_z=max_z)

    # Create summary
    summary_df = create_summary_from_objects(object3D)

    # Draw ellipses
    result = draw_3D_ellipses(
        summary_df, dims=image.shape, label_ellipses=label_ellipses
    )

    return result
