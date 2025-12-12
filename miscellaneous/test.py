from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main
import tifffile
import numpy as np

# One-time setup
Main.include("miscellaneous/OSCARv1.6.1.jl")
Main.eval("using TiffImages")


def run_julia_oscar(labels, zparams=[3, 4, 6]):
    # Save input
    temp_input = "temp_labels.tif"
    tifffile.imwrite(temp_input, labels.astype(np.uint32))

    # Call Julia functions directly
    Main.labels = Main.TiffImages.load(temp_input)
    Main.objc3d, Main.df = Main.ObjSplitter3D(Main.labels, zparams=zparams, nCh=1)
    Main.summ = Main.summary_objs(Main.objc3d)
    Main.oscar_output = Main.draw_3D_ellipses_from_summary(
        Main.summ, dims=Main.size(Main.labels)
    )
    return Main.oscar_output


labels = tifffile.imread(
    r"C:\Users\6331823\Local SSD\Data_Elise\EM_Exp007_Imaging_20x_2025-11-21_Maria_Elise_14.47.50_CZC34578YN_F24\segmented\Frame-10_masks.tif"
)
oscar_output = run_julia_oscar(labels, zparams=[3, 4, 6])
tifffile.imwrite(
    r"C:\Users\6331823\Local SSD\Data_Elise\EM_Exp007_Imaging_20x_2025-11-21_Maria_Elise_14.47.50_CZC34578YN_F24\Frame-10_oscar_output.tif",
    oscar_output,
)
