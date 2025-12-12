using Images

# Load OSCAR functions
include("OSCARv1.6.1.jl")

function run_oscar_segmentation(input_path::String, output_path::String, zparams=[3, 4, 6], nCh=1, kernel = 6.0)
    """
    Run OSCAR segmentation on a label image
    
    Args:
        input_path: Path to input TIFF file with labels (as STRING)
        output_path: Path to save output TIFF file
        zparams: Z-axis parameters for OSCAR [default: [3, 4, 6]]
        nCh: Number of channels [default: 1]
    """
    println("Running OSCAR segmentation with zparams=$zparams, nCh=$nCh, kernel=$kernel")

    image::Union{Matrix{Float64},Array{Float64,3},Array{Int,3},Array{Gray{N0f8},3},Array{Gray{N0f16},3}} = Float64.(load(input_path))

    # image = binarizeImage(image)
    # Images.save("$output_path.binary.tif", Gray.(image))

    image = Int64.(image .> 0)
    
    println("Splitting objects in 3D")
    objc3d, df = ObjSplitter3D_fromBinaryImage(image,zparams,kernel=kernel)
        
    println("Computing summary statistics")
    summ = summary_objs(objc3d)
    
    println("Drawing 3D ellipses")
    # Load image only to get dimensions
    labels = Images.load(input_path)
    oscar_output = draw_3D_ellipses_from_summary(summ, size(image))
    
    println("Saving output to: $output_path")
    oscar_output = Gray.(oscar_output./ maximum(oscar_output))
    Images.save(output_path, oscar_output)
    
    println("OSCAR processing complete!")
    return oscar_output
end

# Parse command line arguments
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia test.jl <input_path> <output_path> [z1] [z2] [z3]")
        exit(1)
    end
    
    input_path = ARGS[1]
    output_path = ARGS[2]
    zparams = length(ARGS) >= 5 ? [parse(Int, ARGS[3]), parse(Int, ARGS[4]), parse(Int, ARGS[5])] : [3, 4, 6]
    
    # Pass STRING paths, not loaded arrays
    run_oscar_segmentation(input_path, output_path, zparams, 1)
end

input_path = "C:/Users/6331823/Local SSD/Data_Elise/EM_Exp007_Imaging_20x_2025-11-21_Maria_Elise_14.47.50_CZC34578YN_F24/segmented/Frame-10_masks_eroded.tif"
output_path = "C:/Users/6331823/Local SSD/Data_Elise/EM_Exp007_Imaging_20x_2025-11-21_Maria_Elise_14.47.50_CZC34578YN_F24/Frame-10_oscar_output_new.tif"

run_oscar_segmentation(input_path, output_path, [3, 5, 9], 1, 6.0)