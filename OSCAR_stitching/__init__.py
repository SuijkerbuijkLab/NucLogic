"""
OSCAR_stitching - A package for stitching 3D objects in organoid segmentation
"""

__version__ = "1.0.0"

# Level 1: Basic utility functions (no internal dependencies)
from .generate_points import generate_points
from .get_covariance import get_covariance
from .linear_regression import linear_regression
from .sort_points_ascending import sort_points_ascending
from .data_2D_chunking import data_2D_chunking

# Level 2: Functions that may use Level 1
from .generate_ellipse_coordinates import generate_ellipse_coordinates
from .fit_3D_line import fit_3D_line
from .sort_points_ascending_map_to_z import sort_points_ascending_map_to_z
from .outliers_detection import outliers_detection

# Level 3: Object handling
from .object_splitter_3D import object_splitter_3D, Objects3D
from .create_summary_from_objects import create_summary_from_objects

# Level 4: Drawing and ellipse operations
from .draw_3D_ellipses import draw_3D_ellipses
from .overlapped_ellipses import overlapped_ellipses
from .pre_obj_elongation import pre_obj_elongation
from .info_preobj_dist_z import info_preobj_dist_z

# Level 5: Higher-level stitching functions
from .ellipses_connector import ellipses_connector
from .terminator_returns import terminator_returns
from .stitch_3D import stitch_3D


# # Define public API
# __all__ = [
#     # Main functions
#     "OSCAR_stitch",
#     "stitch_3D",
#     "object_splitter_3D",
#     "Objects3D",
#     "create_summary_from_objects",
#     "draw_3D_ellipses",
#     # Utility functions
#     "generate_ellipse_coordinates",
#     "generate_points",
#     "get_covariance",
#     "linear_regression",
#     "fit_3D_line",
#     "sort_points_ascending",
#     "sort_points_ascending_map_to_z",
#     "outliers_detection",
#     "data_2D_chunking",
#     # Advanced functions
#     "ellipses_connector",
#     "overlapped_ellipses",
#     "pre_obj_elongation",
#     "info_preobj_dist_z",
#     "terminator_returns",
# ]
