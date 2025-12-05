#=
OSCAR:
- Julia version: 1.10.4
- Author: Míguez Lab.
- Date: 2024-09-24
- Code version: 1.6.1
=#

#Import necessary packages
import Pkg
try
    using IndirectArrays
catch
    Pkg.add("IndirectArrays")
    using IndirectArrays
end
try
    using StatsKit
catch
    Pkg.add("StatsKit")
    using StatsKit
end
try
    using CSV
catch
    Pkg.add("CSV")
    using CSV
end
try
    using Statistics
catch
    Pkg.add("Statistics")
    using Statistics
end
try
    using LinearAlgebra
catch
    Pkg.add("LinearAlgebra")
    using LinearAlgebra
end
try
    using Plots
catch
    Pkg.add("Plots")
    using Plots
end
try
    using DelimitedFiles
catch
    Pkg.add("DelimitedFiles")
    using DelimitedFiles
end
try
    using DataFrames
catch
    Pkg.add("DataFrames")
    using DataFrames
end
try
    using Tables
catch
    Pkg.add("Tables")
    using Tables
end
try
    using Images
catch
    Pkg.add("Images")
    using Images
end
try
    using ImageFiltering
catch
    Pkg.add("ImageFiltering")
    using ImageFiltering
end
try
    using ImageSegmentation
catch
    Pkg.add("ImageSegmentation")
    using ImageSegmentation
end
try
    using ImageFiltering
catch
    Pkg.add("ImageFiltering")
    using ImageFiltering
end
try
    using DataStructures
catch
    Pkg.add("DataStructures")
    using DataStructures
end
try
    using Gtk
catch
    Pkg.add("Gtk")
    using Gtk
end
try
    using Colors
catch
    Pkg.add("Colors")
    using Colors
end
try
    using ProgressBars
catch
    Pkg.add("ProgressBars")
    using ProgressBars
end
try 
    using Dates
catch
    Pkg.add("Dates")
    using Dates
end
#Define functions and structures
"""
    BoundingBox(maxX::Int, minX::Int, maxY::Int, minY::Int)
Structure that contains the minimum and maximum x and y indices of a box in a matrix.
"""
struct BoundingBox
    maxX::Int
    minX::Int
    maxY::Int
    minY::Int
end

"""
    ImageObject(boundingBox::BoundingBox, label::String, inset::Matrix{Int}, mask::Vector{CartesianIndex{2}})
Structure that represents an object of a labeled image. The object with label `label` is constrained within the `bundingBox`.
The `inset` is a `Matrix` containing just the part of the initial image with the object.
The `mask` is a `Vector` of `CartesianIndex` indicating the position of the pixels that the object occupies.
"""
struct ImageObject
    boundingBox::BoundingBox
    label::String
    inset::Matrix{Int}
    mask::Vector{CartesianIndex{2}}
end

"""
    Centroid(x::Float64, y::Float64)
A structure that contains the `x` and `y` coordinates of a `Region`.
"""
struct Centroid
    x::Float64
    y::Float64
end

"""
    Region(label::String, boundingBox::BoundingBox, centroid::Centroid, orientation::Float64, minorAxis::Float64, majorAxis::Float64, area::Float64, matrix::Array{Int})
A structure that contains properties of a labeled `ImageObject`, with label `label`.
"""
struct Region
    label::String
    boundingBox::BoundingBox
    centroid::Centroid
    orientation::Float64
    minorAxis::Float64
    majorAxis::Float64
    area::Float64
    inset::Matrix{Int}
    mask::Vector{CartesianIndex{2}}
end

"""
    HeapItem(value::Float64, age::Int, index::Int, source::Int)
Structure designed for being contained in a `PriorityQueue()` while executing the watershed algorithm.
"""
struct HeapItem
    value::Float64
    age::Int
    index::Int
    source::Int
end

"""
    Pixel(value::Int, age::Int, position::CartesianIndex{2})
This structure represents a pixel in an image, used while watershed algorithm is applied.
"""
struct Pixel
    value::Union{Int,Float64}
    age::Int
    position::CartesianIndex{2}
end

"""
    isless(a::Pixel, b::Pixel)
Custom comparer function for `Pixel`s inside a `PriorityQueue()`.
"""
Base.isless(a::Pixel, b::Pixel) =
    (a.age != b.age) ? (a.age < b.age) : (a.value > b.value)

"""
    isless(a::HeapItem, b::HeapItem)
Custom comparer function for `HeapItem`s inside a `PriorityQueue()`.
"""
Base.isless(a::HeapItem, b::HeapItem) =
    if a.value != b.value
        a.value < b.value
    else
        a.age < b.age
    end

"""
    binarizeImage(image::Union{Matrix{Float64}, Array{Float64, 3}}, threshold::Float64=0.0)::Union{Matrix{Int}, Array{Int, 3}}
Changes all the values greater or equal than threshold in the image by 1.0.
"""
function binarizeImage(image::Union{Matrix{Float64}, Array{Float64, 3}, Array{Gray{Float64},3}}, threshold::Float64=0.0)::Union{Matrix{Int}, Array{Int, 3}}
    return Int.(image.>threshold)
end

"""
    writePropsToFile(file::String, regions::Array{Region})
Creates a CSV file with the information of all `Region`s given. The `file` must contain the whole path of the CSV output (extension not necessary).
"""
function writePropsToFile(file::String, regions::Array{Region})
    if !occursin(".csv", file)
        file *= ".csv"
    end

    str = "Label,Centroid X,Centroid Y,Minor Axis,Major Axis,Angle,Area"

    for r in regions
        str *= "\n" * string(r.label) * "," * string(r.centroid.x) * "," * string(r.centroid.y) * "," * string(r.minorAxis) * "," * string(r.majorAxis) * "," * string(r.orientation) * "," * string(r.area)
    end

    open(file, "w") do file
        write(file, str)
    end
end

"""
    ellipse(r::Int, c::Int, r_radius::Float64, c_radius::Float64, rotation::Float64=0.0, maxSize::Union{Vector{Int}, Nothing}, flipVertically::Bool=true)::Vector{CartesianIndex{2}}
Creates a 2D ellipse with centroid (`r`, `c`) and axis `r_radius` and `c_radius`, rotated `rotation` radians (only between +/- 180º). It returns the `CartesianIndex` of the pixels of an image that the ellipse occupies.
If `flipVertically` is `true`, then the ellipse will be flipped around its vertical axis (like in a mirror.)
# Examples
```
image = zeros(Float64, 600, 600)
ellipseIndices = ellipse(299, 349, 100.0, 220.0, pi/6)
image[ellipseIndices] .= 1.0
```
"""
function ellipse(r::Int, c::Int, r_radius::Float64, c_radius::Float64, rotation::Float64=0.0, maxSize::Union{Vector{Int},Nothing}=nothing, flipVertically::Bool=true)::Vector{CartesianIndex{2}}
    center::Vector{Int} = [r, c]
    radii::Vector{Float64} = [r_radius, c_radius]
    rotation %= pi
    rotation -= pi / 2
    if rotation < 0.0
        rotation += pi
    end
    r_radius_rot::Float64 = abs(r_radius * cos(rotation)) + abs(c_radius * sin(rotation))
    c_radius_rot::Float64 = r_radius * sin(rotation) + abs(c_radius * cos(rotation))
    radii_rot::Vector{Float64} = [r_radius_rot, c_radius_rot]
    upper_left::Vector{Int} = ceil.(Int, center - radii_rot)
    lower_right::Vector{Int} = floor.(Int, center + radii_rot)

    shifted_center::Vector{Int} = center - [i for i in upper_left]
    bounding_shape::Vector{Int} = ([i for i in lower_right] - [i for i in upper_left]) .+ 1
    r_lim::Vector{Int64} = collect(0:bounding_shape[1]-1)
    c_lim::Vector{Int64} = collect(0:bounding_shape[2]-1)
    r_org::Int, c_org::Int = shifted_center
    r_rad::Int, c_rad::Int = floor.(Int, radii)
    rotation %= pi
    sin_alpha::Float64, cos_alpha::Float64 = sin(rotation), cos(rotation)
    r2::Vector{Int} = r_lim .- r_org
    c2::Matrix{Int} = c_lim' .- c_org
    distances::Matrix{Float64} = (((r2 * cos_alpha) .+ (c2 * sin_alpha)) ./ r_rad) .^ 2 + (((r2 * sin_alpha) .- (c2 * cos_alpha)) ./ c_rad) .^ 2
    indices::Vector{CartesianIndex{2}} = findall(distances .< 1)
    rr::Vector{Int64} = [x[1] for x in indices]
    cc::Vector{Int64} = [x[2] for x in indices]
    rr = rr .+ upper_left[1]
    cc = cc .+ upper_left[2]
    indexes = CartesianIndex.(rr, cc)
    if maxSize != nothing
        rem = findall(x -> (x[1] > maxSize[1] || x[2] > maxSize[2] || x[1] <= 0 || x[2] <= 0), indexes)
        indexes = deleteat!(indexes, rem)
    else
        rem = findall(x -> (x[1] <= 0 || x[2] <= 0), indexes)
        indexes = deleteat!(indexes, rem)
    end

    #Do a vertical flip:
    if flipVertically
        indexes .= map(x -> if x[1] > r
            CartesianIndex(x[1] - 2 * (x[1] - r), x[2])
        elseif x[1] < r
            CartesianIndex(x[1] + 2 * (r - x[1]), x[2])
        else
            CartesianIndex(x[1], x[2])
        end, indexes)
    end
    return indexes
end

"""
    padImage(image::Matrix{Float64}, paddingValue::Int=0)::Matrix{Float64}
Adds a 1-pixel border width with value `paddingValue` around an `image`.
"""
function padImage(image::Matrix{Float64}, paddingValue::Int=0)::Matrix{Float64}
    image = hcat(ones(size(image)[1]) .* paddingValue, image, ones(size(image)[1]) .* paddingValue)
    image = vcat(transpose(ones(size(image)[2])) .* paddingValue, image, transpose(ones(size(image)[2])) .* paddingValue)
    return image
end

"""
    removePad(img::Union{Matrix{Int},Matrix{Float64}})::Union{Matrix{Int},Matrix{Float64}}
This function removes a 1-pixel-width border of the `img`.
"""
function removePad(img::Union{Matrix{Int},Matrix{Float64}})::Union{Matrix{Int},Matrix{Float64}}
    return img[2:end-1, 2:end-1]
end

"""
    padImage(image::Matrix{Int}, paddingValue::Int=0)::Matrix{Int}
Adds a 1-pixel border width with value `paddingValue` around an `image`.
"""
function padImage(image::Matrix{Int}, paddingValue::Int=0)::Matrix{Int}
    image = hcat(ones(size(image)[1]) .* paddingValue, image, ones(size(image)[1]) .* paddingValue)
    image = vcat(transpose(ones(size(image)[2])) .* paddingValue, image, transpose(ones(size(image)[2])) .* paddingValue)
    return image
end

"""
    reformatMap(map::Dict)::Dict
It returns a `Dict` which `key`s are the different `values` stored in `map`, and which `values` are the `keys` of `map`. That is: {"15"->[13, 12, 9]} becomes {"13"->[15], "12"->[15], "9"->[15]}.
"""
function reformatMap(map::Dict)::Dict
    for k in keys(map)
        if haskey(map, k)
            array = map[k]
            x = 1
            len = length(array)

            while x <= len
                c = array[x]
                if haskey(map, string(c))
                    array2 = map[string(c)]
                    array = vcat(array, array2)
                    map[k] = array
                    x = 1
                    len = length(array)
                    delete!(map, string(c))
                else
                    x = x + 1
                end
            end
        end
    end
    nMap::Dict = Dict()
    for k in keys(map)
        values = map[k]
        for v in values
            nMap[string(v)] = parse(Float64, k)
        end
    end
    return nMap
end

"""
    findObjects(image::Matrix{Int}, minSizeX::Int=2, minSizeY::Int=2)::Array{ImageObject}
Finds the objects of a labeled image which background value is `0` and the labels are positive `Int` values. Returns an array with all the `ImageObject`s found that are bigger than `minSizeX`x`minSizeY` pixels.
# Examples
```
image = zeros(Int, 600, 600)
ellipseIndices = ellipse(299, 349, 100.0, 220.0, pi/6)
image[ellipseIndices] .= 1
objects = findObjects(labelBinaryImage(image))
```
"""
function findObjects(labeledImage::Matrix{Int}, minSizeX::Int=1, minSizeY::Int=1)::Array{ImageObject}
    if minSizeX < 1 || minSizeY < 1
        throw("Filter is too little! Minimum value accepted for minSizeX and minSizeY is 1")
    end
    #Obtain the different-than-0 values of the image.
    #The length of this array is the number of objects
    values::Array{Int} = setdiff!(unique(labeledImage), 0)

    objects::Array{ImageObject} = []

    #Iterate in all the different labels
    #Threads.@threads :dynamic
    for v in values
        #Get the points of the object
        indexes::Vector{CartesianIndex{2}} = findall(x -> x == v, labeledImage)

        #Get the coordinates of the object in x and y
        xCoords::Vector{Int} = [x[1] for x in indexes]
        yCoords::Vector{Int} = [y[2] for y in indexes]

        #Filter too little objects (less than 2x2)
        box = BoundingBox(
            maximum(xCoords),
            minimum(xCoords),
            maximum(yCoords),
            minimum(yCoords)
        )

        #Only count objects with size > (minSizeX, minSizeY)
        if (box.maxX - box.minX) > minSizeX && (box.maxY - box.minY) > minSizeY
            #Create inset
            matrix::Matrix{Int} = labeledImage[
                box.minX:box.maxX,
                box.minY:box.maxY
            ]

            #Create mask
            #mask::Matrix{Int} = zeros(Int, size(labeledImage))
            #mask[indexes] .= 1

            #Push the object to the return array
            push!(objects,
                ImageObject(box, string(v), matrix, indexes)
            )
        end

    end

    return objects
end

"""
    inertiaTensor(image::Matrix{Int}, mu::Matrix{Float64}, spacing::Matrix{Int})::Matrix{Float64}
Calculates the inertia tensor for a 2D central moments matrix.
"""
function inertiaTensor(mu::Matrix{Float64}, spacing::Matrix{Int}=[1 1])::Matrix{Float64}
    corners2::Vector{CartesianIndex{2}} = [CartesianIndex(1, 3), CartesianIndex(3, 1)]
    result::Matrix{Float64} = diagm((sum(mu[corners2]) .- mu[corners2]) / mu[1, 1])
    result[[CartesianIndex(1, 2), CartesianIndex(2, 1)]] .= -mu[CartesianIndex(2, 2)] / mu[1, 1]
    return result
end

"""
    momentsCentral(image::Matrix{Int}, order::Int=3, center::Union{Matrix{Float64}, Nothing}=nothing, spacing::Union{Matrix{Int}, Nothing}=nothing)::Matrix{Float64}
Calculates the central moments of an `image` up to a certain `order`.
"""
function momentsCentral(image::Matrix{Int}, order::Int=2, center::Union{Matrix{Float64},Nothing}=nothing, spacing::Union{Matrix{Int},Nothing}=nothing)::Matrix{Float64}
    calc::Matrix{Float64} = image

    if center == nothing
        center = [0.0 0.0]
    end

    if spacing == nothing
        spacing = ones(size(image))
    end

    for dim = 1:ndims(image)
        delta::Vector{Float64} = [i for i in (1:size(image, dim)) .- 1] .* spacing[dim] .- center[dim]
        powers_of_delta::Matrix{Float64} = transpose(delta) .^ [i for i in ((1:order+1) .- 1)]
        if dim == 1
            calc = transpose(calc)
        end
        calc = calc * transpose(powers_of_delta)
        calc = transpose(calc)
    end

    return copy(calc)
end

"""
    labelBinaryImage(binaryImage::Matrix{Int})::Matrix{Int}
From a `binaryImage` (`0` background and `1` signal), finds all the connected pixels and changes their values by a positive `Int` (not necessarily in sequential order).
# Examples
```
image = zeros(Int, 600, 600)
ellipseIndices = ellipse(299, 349, 100.0, 220.0, pi/6)
image[ellipseIndices] .= 1
labeledImage = labelBinaryImage(image)
```
"""
function labelBinaryImage(binaryImage::Matrix{Int})::Matrix{Int}
    #Get image size
    h::Int = size(binaryImage)[1]
    w::Int = size(binaryImage)[2]

    #Add a 0-border padding
    img::Matrix{Float64} = padImage(binaryImage)

    #First tag
    tag::Int = 1

    #Map with tricky labels
    map::Dict = Dict()

    #First pass
    for i in 2:h+1
        for j in 2:w+1

            if img[i, j] == 1
                #Find neighbours
                neighbours::Matrix{Int} = [img[i, j-1] img[i-1, j-1] img[i-1, j] img[i-1, j+1]]

                #Find non-zero neighbours
                nzn::Vector{CartesianIndex{2}} = findall(x -> x != 0, neighbours)

                #All neighbours are zero: new element
                if length(nzn) == 0
                    img[i, j] = tag
                    tag = tag + 1

                    #One neighbour is non-zero: set that element's tag
                elseif length(nzn) == 1
                    img[i, j] = neighbours[nzn][1]
                    #More than one neighbour is non-zero
                else
                    #Set lowest label of all neighbours
                    l::Float64 = minimum([neighbours[i] for i in nzn[:]])
                    img[i, j] = minimum([neighbours[i] for i in nzn[:]])

                    #Record equivalence classes
                    for k in nzn
                        templ::Float64 = neighbours[k][1]
                        if templ != l
                            if haskey(map, string(l))
                                map[string(l)] = unique([map[string(l)]' templ])
                            else
                                map[string(l)] = templ
                            end
                        end
                    end
                end
            end
        end
    end

    #Restructure map
    map = reformatMap(map)

    #Remove padding
    img = img[2:h+1, 2:w+1]

    #Second pass (set proper labels)
    for i in 1:h
        for j in 1:w
            if haskey(map, string(img[i, j]))
                img[i, j] = map[string(img[i, j])]
            end
        end
    end

    return [floor(Int, x) for x in img]
end


"""
    drawKernel(size::Int, r::Float64, removeCenter::Bool = false)::Matrix{Int}
Returns a square 2D-matrix of size size with all its values set to 0 except for those
that fall inside the kernel of radius r.
If removeCenter is true, value of center pixel will be 0.
#Examples
r = drawKernel(5, 2.0)
heatmap(r,aspect_ratio=:equal)
"""
function drawKernel(size::Int, r::Float64, removeCenter::Bool=false)::Matrix{Int}
    base::Matrix{Int} = Int.(zeros(size, size))
    center::Int = Int.(ceil(size / 2))
    for i in 1:size
        for j in 1:size
            if sqrt((i - center)^2 + (j - center)^2) <= (r + 0.5)
                base[i, j] = 1
            end
        end
    end
    if removeCenter
        base[center, center] = 0
    end
    return base
end

"""
    drawKernel(r::Float64, removeCenter::Bool = false)::Matrix{Int}
Returns a square 2D-matrix of auto-size with all its values set to 0 except for those
that fall inside the kernel of radius r. Works in a similar way as Fiji's kernel creator.
If removeCenter is true, value of center pixel will be 0.
#Examples
r = drawKernel(5, 2.0)
heatmap(r,aspect_ratio=:equal)
"""
function drawKernel(r::Float64, removeCenter::Bool=false)
    size::Int = 0
    if r>=2.5 && r<3
        size = 7
    elseif r<2.5
        size = ceil(r)*2+1
    else
        if (ceil(r)==r)
            size = ceil(r+1)*2-1
        else
            size = ceil(r)*2-1 
        end
    end

    base::Matrix{Int} = Int.(zeros(size, size))
    center::Int = Int.(ceil(size / 2))
    for i in 1:size
        for j in 1:size
            if sqrt((i - center)^2 + (j - center)^2) <= (r + 0.5)
                base[i, j] = 1
            end
        end
    end
    if removeCenter
        base[center, center] = 0
    end
    return base, size
end


"""
    removeEqualNeighbours(img::Matrix{Float64}, size::Int = 5, r::Float64 = 2.5)::Matrix{Float64}

Removes pixels in an image that have the same value as their neighboring pixels within a specified radius.

**Arguments:**
- `img`: The input image as a 2D matrix of floating-point values.
- `size`: The size of the square neighborhood to consider around each pixel. Defaults to 5.
- `r`: The radius of the circular neighborhood within the square neighborhood defined by `size`. Defaults to 2.5.

**Returns:**
- The modified image with removed equal neighboring pixels.

**Example:**
```julia
# Load an image
img = load("image.jpg")

# Remove equal neighbors with a neighborhood size of 7 and a radius of 3
modified_img = removeEqualNeighbours(img, size=7, r=3)
"""
function removeEqualNeighbours(img::Matrix{Float64}, r::Float64=2.5)::Matrix{Float64}
    kernel, sizeBox = drawKernel(r, true) 
    for i in 2:((sizeBox-1)/2)
        img = padImage(img)
    end
    pixelList::Vector{CartesianIndex{2}} = findall(img .!= 0)
    pixelsToMod::Vector{CartesianIndex{2}} = []
    meanNeighbourhoodValue::Vector{Float64} = []
    for element in pixelList
        if !(element in pixelsToMod)
            neighboursList = findall(kernel .== 1) .- CartesianIndex(Int.(ceil(sizeBox / 2)), Int.(ceil(sizeBox / 2))) .+ element
            equalPixels::Vector{Int} = findall(img[neighboursList] .== img[element])
            if length(equalPixels) == 1
                push!(pixelsToMod, neighboursList[equalPixels[1]])
                push!(meanNeighbourhoodValue, mean(img[neighboursList]))
            end
        end
    end
    img[pixelsToMod] .= meanNeighbourhoodValue
    for i in 2:((sizeBox-1)/2)
        img = removePad(img)
    end
    return img
end

"""
    findmaxima_inHouse(img::Matrix{Float64})

Find pixels considered as local maxima in the image.

**Arguments:**
- `img`: The input image as a 2D matrix of floating-point values.

**Returns:**
- A vector of CartesianIndex containing maxima positions.

**Example:**
```julia
# Load an image
img = load("image.jpg")

# Find image local maxima
maxima = findmaxima_inHouse(img)
"""
function findmaxima_inHouse(img::Matrix{Float64})
    try
        #Pad image to avoid put-of-index errors
        img = padImage(img, 0)
        # println(size(img[findall(img.>0)]))
        meanDist = maximum(img[findall(img.>0)])
        #Find all pixels that are !=0 (won't be any max there)
        coords::Vector{CartesianIndex{2}} = findall(img .> 0.5*meanDist)

        #Priority queue
        pq::PriorityQueue = PriorityQueue()

        #List to store maximum coords
        maximaCoord::Vector{CartesianIndex{2}} = []

        #Enqueue pixels
        for i in coords
            pixel::Pixel = Pixel(img[i], 0, i)
            enqueue!(pq,pixel,pixel)
        end

        while length(pq) > 0
            p::Pixel = dequeue!(pq)

            # #
            # neighboursList::Vector{CartesianIndex{2}} = [
            #     p.position .+ CartesianIndex(1, 0),
            #     p.position .+ CartesianIndex(0, 1),
            #     p.position .- CartesianIndex(1, 0),
            #     p.position .- CartesianIndex(0, 1),
            #     p.position .+ CartesianIndex(1, 1),
            #     p.position .+ CartesianIndex(-1, -1),
            #     p.position .+ CartesianIndex(1, -1),
            #     p.position .+ CartesianIndex(-1, 1)
            # ]
            kernel, sizeBox = drawKernel(1.0, true)
            neighboursList = findall(kernel .== 1) .- CartesianIndex(Int.(ceil(sizeBox / 2)), Int.(ceil(sizeBox / 2))) .+ p.position

            greaterPixels::Vector{Int} = findall(img[neighboursList] .> img[p.position])
            # equalPixels::Vector{Int} = findall(img[neighboursList] .== img[p.position])
            if length(greaterPixels) == 0 
                push!(maximaCoord,p.position)
                kernel, sizeBox = drawKernel(p.value, true)
                pixelsToRemove = findall(kernel .== 1) .- CartesianIndex(Int.(ceil(sizeBox / 2)), Int.(ceil(sizeBox / 2))) .+ p.position
                for n in pixelsToRemove
                    pn = Pixel(img[n], 0, n)
                    try
                        delete!(pq,pn)
                    catch e
                    end
                end
            end 
        end
        maximaCoord = maximaCoord .- CartesianIndex(1,1)
        return maximaCoord
    catch e 
        return []
    end
end

"""
    findObjectsWatershed(binaryImage::Matrix{Int}, useInHouseAlgorithm::Bool=true, returnLabeledImage::Bool=false, excludeBorder::Bool=false;kernel::Float64=3.0 )::Array{ImageObject}
Finds the objects in a `binaryImage` (0: background, 1: signal) applying a watershed algorithm to separate overlapping objects. It returns the array of `ImageObject`s found by the algorithm.
If watershed algorithm fails, then the basic `findObjects` function is used to create the `ImageObject`s array that is returned, applying a simple labeling of unconnected regions.
If `returnLabeledImage` is `true`, then the labeled image is returned instead of the `ImageObject`s array.
If `excludeBorder` is true, then objects touching the border of the image will not be counted.
If `useInHouseAlgorithm` is `true`, then the custom watershed algorithm will be used.
# Examples
```
image = zeros(Int, 80, 80)
ellipseIndices1 = ellipse(29, 29, 16.0, 16.0)
ellipseIndices1 = ellipse(45, 53, 20.0, 20.0)
image[ellipseIndices1] .= 1
image[ellipseIndices2] .= 1
objects = findObjectsWatershed(image)
```
"""
function findObjectsWatershed(
    binaryImage::Matrix{Int},
    useInHouseAlgorithm::Bool=true,
    returnLabeledImage::Bool=false,
    excludeBorder::Bool=true;
    kernel::Float64=6.0
)::Union{Array{ImageObject}, Matrix{Int}}

    if !useInHouseAlgorithm
        # --- NON-IN-HOUSE ALGORITHM (Optimizations applied) ---
        # This path contains a from-scratch watershed implementation.
        # Key optimizations:
        # 1. Avoid `try/catch` in the hot loop.
        # 2. Use CartesianIndices for neighbor checks to avoid manual index calculation and bounds errors.

        logImage = .!Bool.(binaryImage)
        dists = distance_transform(feature_transform(logImage))

        # Create a copy to modify if we need to exclude borders
        local_binaryImage = copy(binaryImage)
        if excludeBorder
            local_binaryImage[1, :] .= 0
            local_binaryImage[:, end] .= 0
            local_binaryImage[:, 1] .= 0
            local_binaryImage[end, :] .= 0
        end

        try
            # OPTIMIZATION: Use views to avoid copying large chunks of the distance map.
            box = findObjects(local_binaryImage)[1].boundingBox
            imgObj_view = @view dists[box.minX:box.maxX, box.minY:box.maxY]

            imgMask = imgObj_view .> 0

            # Create a mutable copy to work with
            imgObj = copy(imgObj_view)
            imgObj[.!imgMask] .= typemin(Float64)

            # OPTIMIZATION: Use a dedicated local maxima function if available,
            # as it's typically faster than `mapwindow`.
            # This is a placeholder for a potentially more optimized maxima finder.
            peakMask = findlocalmaxima(imgObj)
            peakMask .&= (imgObj .> 0)

            # Coordinates of the max peaks
            peakCoords = findall(peakMask)
            # Adjust coordinates back to the full image space
            peakCoords_global = [CartesianIndex(ci[1] + box.minX - 1, ci[2] + box.minY - 1) for ci in peakCoords]

            markers = zeros(Int, size(dists))
            for (i, coord) in enumerate(peakCoords_global)
                markers[coord] = i
            end

            # Watershed implementation
            # Using a priority queue with (priority, data)
            queue = PriorityQueue{CartesianIndex{2}, Float64}()

            # Invert distances so higher values are processed first (lower priority)
            inverted_dists = -dists
            labeled_image = copy(markers)

            for coord in findall(x -> x > 0, markers)
                enqueue!(queue, coord, inverted_dists[coord])
            end

            # OPTIMIZATION: Pre-calculate neighbor offsets instead of creating a new array for each pixel.
            neighbor_offsets = [CartesianIndex(0, 1), CartesianIndex(0, -1), CartesianIndex(1, 0), CartesianIndex(-1, 0)]

            while !isempty(queue)
                current_idx = dequeue!(queue)
                current_label = labeled_image[current_idx]

                for offset in neighbor_offsets
                    neighbor_idx = current_idx + offset
                    
                    # OPTIMIZATION: Use `checkbounds` instead of try/catch for performance.
                    if checkbounds(Bool, labeled_image, neighbor_idx)
                        if local_binaryImage[neighbor_idx] != 0 && labeled_image[neighbor_idx] == 0
                            labeled_image[neighbor_idx] = current_label
                            enqueue!(queue, neighbor_idx, inverted_dists[neighbor_idx])
                        end
                    end
                end
            end

            return returnLabeledImage ? labeled_image : findObjects(labeled_image)

        catch e
            @warn("Watershed algorithm failed, using basic image segmentation.", e)
            labeled = labelBinaryImage(binaryImage)
            return returnLabeledImage ? labeled : findObjects(labeled)
        end
    else
        # --- IN-HOUSE ALGORITHM (Optimizations applied) ---
        kernel_op = Bool.(drawKernel(kernel / 2)[1])
        binaryImg = opening(binaryImage, kernel_op)

        # Pad the image once for all operations
        pImage = padImage(binaryImg, 0)
        
        distances = distance_transform(feature_transform(.!Bool.(pImage)))
        distances[.!Bool.(pImage)] .= typemin(Float64)

        maximumCoords = findmaxima_inHouse(distances)
        outImage = zeros(Int, size(pImage))
        
        # Use a PriorityQueue that prioritizes pixels with higher values (closer to centers)
        # Storing (value, age, position) where value is from the distance map.
        q = PriorityQueue{CartesianIndex{2}, Tuple{Float64, Int}}()
        age = 1

        for (i, coord) in enumerate(maximumCoords)
            outImage[coord] = i
            # Enqueue with negative distance as priority (so max distance is highest priority)
            # and age as a tie-breaker.
            enqueue!(q, coord, (-distances[coord], age))
            age += 1
        end
        
        # OPTIMIZATION: Pre-calculate neighbor offsets.
        neighbor_offsets = [CartesianIndex(1, 0), CartesianIndex(-1, 0), CartesianIndex(0, 1), CartesianIndex(0, -1)]

        while !isempty(q)
            pixel_pos = dequeue!(q)
            pixel_value = outImage[pixel_pos]

            for offset in neighbor_offsets
                n_pos = pixel_pos + offset
                # OPTIMIZATION: Combined check. No need for `checkbounds` as `pImage` is padded.
                if pImage[n_pos] != 0 && outImage[n_pos] == 0
                    outImage[n_pos] = pixel_value
                    
                    age += 1
                    enqueue!(q, n_pos, (-distances[n_pos], age))
                end
            end
        end

        _labeledImage = removePad(outImage)
        return returnLabeledImage ? _labeledImage : findObjects(_labeledImage)
    end
end

"""
    regionProps(objects::Array{ImageObject})::Array{Region}
From an `ImageObject`s array, it returns an array of `Region`s with all the properties of each `Region` in the image.
# Examples
```
image = zeros(Int, 600, 600)
ellipseIndices = ellipse(299, 349, 100.0, 220.0, pi/6)
image[ellipseIndices] .= 1
regions = regionProps(findObjects(labelBinaryImage(image)))
```
"""
function regionProps(objects::Array{ImageObject})::Array{Region}
    # OPTIMIZATION: Initialize an empty vector and push to it.
    # This avoids pre-allocation and the need to filter unassigned elements later.
    regions = Region[]
    
    for o in objects
        # OPTIMIZATION: Avoid creating intermediate arrays with `findall`.
        # Calculate centroid by iterating directly.
        sum_x, sum_y, area = 0.0, 0.0, 0
        
        # Iterate over the bounding box of the inset
        for ci in CartesianIndices(o.inset)
            if o.inset[ci] >= 1
                # Sum coordinates in the original image space
                sum_y += ci[1] + o.boundingBox.minX - 1
                sum_x += ci[2] + o.boundingBox.minY - 1
                area += 1
            end
        end

        if area == 0 continue end

        centroid = Centroid(sum_x / area, sum_y / area)

        # OPTIMIZATION: Calculate moments with a single pass.
        # This avoids creating a temporary binary matrix.
        μ20, μ02, μ11 = 0.0, 0.0, 0.0
        centroid_local_x = centroid.x - (o.boundingBox.minY - 1)
        centroid_local_y = centroid.y - (o.boundingBox.minX - 1)

        for ci in CartesianIndices(o.inset)
             if o.inset[ci] >= 1
                dx = (ci[2] - centroid_local_x)
                dy = (ci[1] - centroid_local_y)
                μ20 += dx^2
                μ02 += dy^2
                μ11 += dx * dy
            end
        end
        μ20 /= area
        μ02 /= area
        μ11 /= area

        iT = [μ20 -μ11; -μ11 μ02] # Inertia Tensor

        orientation = 0.5 * atan(-2 * iT[1, 2], iT[2, 2] - iT[1, 1])

        orientation *= 180 / pi
        orientation += 90
        if orientation >= 180
            orientation -= 180
        end

        try
            eigVals = eigvals(iT)
            # Ensure order for major/minor axis calculations
            sort!(eigVals) 
            
            # Using sqrt(max(0, val)) to avoid DomainError with slightly negative eigenvalues
            # from numerical precision issues.
            minor_axis = 4 * sqrt(max(0.0, eigVals[1]))
            major_axis = 4 * sqrt(max(0.0, eigVals[2]))
            
            # OPTIMIZATION: Push the new Region object to the dynamically growing array.
            push!(regions, Region(
                o.label, o.boundingBox, centroid, orientation,
                minor_axis, major_axis,
                pi * major_axis * minor_axis / 4, # Area of fitted ellipse
                o.inset .>= 1, # Pass the boolean mask directly
                o.mask
            ))
        catch e
            # Handle potential numerical errors, e.g., in eigen decomposition
            @warn "Could not process region $(o.label) due to: $e"
        end
    end
    # OPTIMIZATION: Simply return the populated array. No filtering needed.
    return regions
end

"""
    regionPropsStack(image::Array{Int, 3}, cartesianAxis::String, useWatershed::Bool=false, useInHouseAlgorithm::Bool=true; kernel::Float64 = 3.0)::Vector{Array{Region}}
From an image of 3 dimensions, it labels each of the slices, finds the objects as returns a `Vector` containing `Arrays` (one per slice), each of them with the `Region`s found in that slice.
If `useWatershed` is `true`, then a watershed segmentation algorithm will be used before labelling the image.
If `useInHouseAlgorithm` is `true`, a custom watershed algorithm will be used instead of the skimage's one.
The `cartesianAxis` can either be "x", "y" or "z", and it indicates along which axis the slices will be taken.
For an image of 20 slices, each of them 4x5, the length of the return `Vector` will be:
    · "x" -> a
    · "y" -> 5
    · "z" -> 20
# Examples
```
img = Int.(load("test.tif"))
regions = regionPropsStack(img, "x")
```
"""
function regionPropsStack(image::Array{Int,3}, cartesianAxis::String, useWatershed::Bool=false, useInHouseAlgorithm::Bool=true; kernel::Float64 = 6.0)::Vector{Array{Region}}

    axis = (cartesianAxis == "z") ? 3 : ((cartesianAxis == "y") ? 1 : 2)
    num_slices = size(image, axis)
    regs = Vector{Array{Region}}(undef, num_slices)

    # OPTIMIZATION: Use multithreading to process slices in parallel.
    # This provides a significant speedup on multi-core machines.
   Threads.@threads for i in tqdm(1:num_slices)
        # OPTIMIZATION: Use a view (`@view`) to avoid allocating a new matrix for each slice.
        # This is a major performance improvement.
        _image = if axis == 3
            @view image[:, :, i]
        elseif axis == 1
            @view image[i, :, :]
        else
            @view image[:, i, :]
        end
        
        # Convert view to a concrete array if the downstream function requires it.
        # Many functions in Julia's ecosystem work fine with views.
        slice_data = collect(_image)

        if useWatershed
            regs[i] = regionProps(findObjectsWatershed(slice_data, useInHouseAlgorithm, kernel=kernel))
        else
            regs[i] = regionProps(findObjects(labelBinaryImage(slice_data)))
        end
        # OPTIMIZATION: Removed the manual `GC.gc()` call. By reducing allocations
        # with views and optimized loops, manual garbage collection is unnecessary and
        # can even hurt performance by pausing execution.
    end

    return regs
end

"""
    generateEllipsesImage(regions::Vector{Array{Region}}, sliceDims::Tuple{Int, Int})::Array{Int, 3}
Given a `Vector` of length `z` (`regions`) and the dimensions `sliceDims` of each slice, this functions creates an stack image of `z` slices of size `sliceDims`.
In each slice `i`, the function draws the ellipses stored in the `Array` of `Region`s in `regions[i]`.
"""
function generateEllipsesImage(regions::Vector{Array{Region}}, sliceDims::Tuple{Int,Int})::Array{Int,3}
    ellipseImage = []

    for z in tqdm(1:length(regions))
        slice = zeros(Int, sliceDims)
        Threads.@threads for r in regions[z]
            #Optimized to get the same output as Fiji
            e = ellipse(floor(Int, r.centroid.x), floor(Int, r.centroid.y), r.minorAxis / 2, r.majorAxis / 2, r.orientation * pi / 180)
            e2 = e[findall(x -> (x[1] <= sliceDims[1] && x[2] <= sliceDims[2]), e)]
            e2 = deleteat!(e2, findall(x -> (x[1] <= 0 || x[2] <= 0), e2))
            slice[e2] .= 1
        end
        push!(ellipseImage, slice)
        GC.gc()
    end

    return cat(ellipseImage..., dims=3)
end

"""
    min_max_Zdef_fromJuliaRegionProps(regions::Vector{Array{Region}}; medZ=0)

Calculates minimum, median, and maximum Z values from a vector of JuliaRegionProps objects.

**Arguments:**
* `regions`: A vector of arrays, where each array contains JuliaRegionProps objects.
* `medZ`: Optional user-defined median Z value.

**Returns:**
A `zParam` struct containing the calculated minimum, median, and maximum Z values.

**Functionality:**
1. Extracts bounding box heights (`bbW`) and areas (`areaW`) from the `regions` data.
2. Calculates minimum, median, and maximum Z values based on `bbW` and optionally the provided `medZ`.
3. Returns a `zParam` struct containing the calculated values.

**Note:**
* The function assumes a specific structure for the `JuliaRegionProps` objects, including properties like `boundingBox` and `area`.
* The logic for calculating minimum, median, and maximum Z values is similar to the original `min_max_Zdef` function.
* The function could be optimized by pre-allocating arrays for `areaW` and `bbW` to improve performance.
"""
function min_max_Zdef_fromJuliaRegionProps(regions::Vector{Array{Region}}; medZ=0)
    #
    #
    #
    #
    areaW = [item for sublist in [[y.area for y in x] for x in regions] for item in sublist]

    bbW = [item for sublist in [[y.boundingBox.maxY - y.boundingBox.minY for y in x] for x in regions] for item in sublist]
    # indexX = 3; indexY = 4; indexArea = 2; indexZ = 1; indexAng = 5; indexA = 6; indexB = 7;
    # indexFeret = 8; indexFeretX = 9; indexFerety = 10; indexFeretAngle = 11; indexFeretmin = 12;
    # indexBX = 13; indexBY= 14;indexArea = 2; indexBW= 15; indexBH= 16;
    # abW = data[:,indexBW]; bbW  = data[:,indexBH];
    if medZ > 0
        var2 = [bbW[i] for i in 1:length(bbW) if medZ < bbW[i]]
        if length(var2) > 0
            maxZ = mean(var2)
        else
            maxZ = medZ
        end
        var2 = [bbW[i] for i in 1:length(bbW) if medZ > bbW[i]]
        if length(var2) > 0
            minZ = mean(var2)
        else
            minZ = medZ
        end
    else
        # indV = [i for i in 1:length(bbW) if percentile(bbW, 25)<bbW[i]<percentile(bbW,75)];
        # areaW = areaW[indV]; bbW = bbW[indV];
        # value1 = mean(areaW);
        # var2 = [bbW[i] for i in 1:length(areaW) if value1<areaW[i]]; medZ = mean(var2);
        value1 = mean(areaW)
        var2 = [bbW[i] for i in 1:length(areaW) if value1 < areaW[i]]
        value1 = mean(var2)
        medZ = (round(value1))
        # var2 = [bbW[i] for i in 1:length(bbW) if value1<bbW[i]]; medZ = mean(var2);
        # medZ = (round(medZ));
        var2 = [bbW[i] for i in 1:length(bbW) if medZ < bbW[i]]
        if isempty(var2) == false
            maxZ = mean(var2)
        else
            maxZ = medZ
        end
        # maxZ = mean(var2);
        var2 = [bbW[i] for i in 1:length(bbW) if medZ > bbW[i]]
        if isempty(var2) == false
            minZ = mean(var2)
        else
            minZ = medZ
        end
    end
    #
    minZ = Int(round(minZ))
    medZ = Int(round(medZ))
    maxZ = Int.(round(maxZ))
    return (zParam(minZ, medZ, maxZ))
end

"""
    data2d_fromJuliaRegionProps(regions, nCh)

Converts a vector of JuliaRegionProps objects to a 2D array format.

**Arguments:**
* `regions`: A vector of JuliaRegionProps objects, likely obtained from image processing.
* `nCh`: Number of additional channels to include in the output data.

**Returns:**
* `data2d`: A vector of 2D arrays, where each array represents a slice of object data.

**Functionality:**
1. Initializes an empty vector `data2d` to store the output data.
2. Finds indices of non-empty regions in the `regions` vector.
3. Iterates over the non-empty regions:
   * Creates a 2D array `Zi` to store data for the current region.
   * Iterates over each region in the current slice:
     * Extracts relevant properties from the region object and stores them in `Zi`.
     * Initializes additional channels with zeros (placeholder for future implementation).
   * Appends `Zi` to the `data2d` vector.
4. Returns the `data2d` vector.

**Note:**
* The function assumes a specific structure for the `JuliaRegionProps` objects.
* The additional channels are currently filled with zeros and might need to be populated with actual data.
* The output format is designed to be compatible with other functions in the codebase.
"""
function data2d_fromJuliaRegionProps(regions, nCh)
    data2d = Vector{Array{Float64}}()
    nonEmptyRegs = findall(length.(regions) .!= 0)
    counter = 1
    for i in nonEmptyRegs
        Zi = zeros(length(regions[i]), 11 + nCh)
        j = 1
        for r in regions[i]
            Zi[j, 1] = r.centroid.x #Xc
            Zi[j, 2] = r.centroid.y #Yc
            Zi[j, 3] = r.area #Area
            Zi[j, 4] = r.orientation #Angle
            Zi[j, 5] = i #Z
            Zi[j, 6] = r.majorAxis #Major exe
            Zi[j, 7] = r.minorAxis #Minor exe
            Zi[j, 8] = 0 #Tag of object
            Zi[j, 9] = j #Index in Zdata
            Zi[j, 10] = counter #This values represent the position in the vector of matrices which compose data2d, its to define to which index in data2d you should acces to get the value.
            Zi[j, 11] = 0 #Index in Data; lets try masks here.
            for k in 1:nCh
                Zi[j, 11+k] = 0 #Here will be the channels intensity values FUTURE WORK
            end
            j += 1
        end
        push!(data2d, Zi)
        counter +=1
    end
    return data2d
end

"""
    zParam(minZ::Int, medZ::Int, maxZ::Int)

Structure containing minimum, medium, and maximum Z values.

Fields:
* `minZ`: Minimum Z value.
* `medZ`: Medium Z value.
* `maxZ`: Maximum Z value.
"""
struct zParam
    minZ::Int
    medZ::Int
    maxZ::Int
end

"""
    overlappedInfo(xcoordA::Vector{Float64}, ycoordA::Vector{Float64}, xcoordB::Vector{Float64}, ycoordB::Vector{Float64}, AonB::Int, BonA::Int)

Structure containing overlap information between two ellipses.

Fields:
* `xcoordA`: X coordinates of points in ellipse A.
* `ycoordA`: Y coordinates of points in ellipse A.
* `xcoordB`: X coordinates of points in ellipse B.
* `ycoordB`: Y coordinates of points in ellipse B.
* `AonB`: Number of points from ellipse A falling into ellipse B.
* `BonA`: Number of points from ellipse B falling into ellipse A.
"""
struct overlappedInfo
    xcoordA::Vector{Float64}
    ycoordA::Vector{Float64}
    xcoordB::Vector{Float64}
    ycoordB::Vector{Float64}
    AonB::Int
    BonA::Int
end

"""
    Line(beta::Vector{Float64}, ypredicted::Vector{Float64}, y::Vector{Float64}, supCI::Vector{Float64}, downCI::Vector{Float64}, Rsquared::Float64)

Structure representing a regression line.

Fields:
* `beta`: Regression coefficients.
* `ypredicted`: Predicted y values.
* `y`: Original y values.
* `supCI`: Upper confidence interval.
* `downCI`: Lower confidence interval.
* `Rsquared`: R-squared value.
"""
struct Line
    beta::Vector{Float64}
    ypredicted::Vector{Float64}
    y::Vector{Float64}
    supCI::Vector{Float64}
    downCI::Vector{Float64}
    Rsquared::Float64
end

"""
    Vector3D(xpred::Vector{Float64}, ypred::Vector{Float64}, xcenter::Float64, ycenter::Float64, zcenter::Float64, xref::Float64, yref::Float64, betaX::Vector{Float64}, betaY::Vector{Float64}, vx::Float64, 
    vy::Float64, vz::Float64, vmodule::Float64)

Structure representing a 3D vector with additional information.

Fields:
* `xpred`: Predicted x values.
* `ypred`: Predicted y values.
* `xcenter`: X coordinate of the center.
* `ycenter`: Y coordinate of the center.
* `zcenter`: Z coordinate of the center.
* `xref`: Reference x value in the next plane.
* `yref`: Reference y value in the next plane.
* `betaX`: Regression coefficients for x.
* `betaY`: Regression coefficients for y.
* `vx`: x component of the vector.
* `vy`: y component of the vector.
* `vz`: z component of the vector.
* `vmodule`: Magnitude of the vector.
"""
struct Vector3D
    xpred::Vector{Float64}
    ypred::Vector{Float64}
    xcenter::Float64
    ycenter::Float64
    zcenter::Float64
    xref::Float64 # Predicted point in the next plane
    yref::Float64
    betaX::Vector{Float64}
    betaY::Vector{Float64}
    vx::Float64
    vy::Float64
    vz::Float64
    vmodule::Float64
end

"""
    infoPreobj(dist::Vector{Float64}, ang::Vector{Float64}, pearson::Float64)

Structure containing information about a pre-object.

Fields:
* `dist`: Vector of distances from ellipse centroids to the 3D line center.
* `ang`: Vector of angles between the 3D line and subsequent ellipsoid centroids.
* `pearson`: Pearson correlation coefficient.
"""
struct infoPreobj
    dist::Vector{Float64}
    ang::Vector{Float64}
    pearson::Float64
end

"""
    jointEllipses_output(nextEl_2::Union{Int,Nothing}, nextStep::Int, cx::Float64, cy::Float64)

Structure containing information for pre-object elongation progression.

Fields:
* `nextEl_2`: Index of the next selected ellipse.
* `nextStep`: Binary variable indicating if elongation continues.
* `cx`: Projected x coordinate for searching the closest ellipse in the next slice.
* `cy`: Projected y coordinate for searching the closest ellipse in the next slice.
"""
struct jointEllipses_output
    nextEl_2::Union{Int,Nothing}
    nextStep::Int
    cx::Float64 # Projected point to search nearest points by Line_3D
    cy::Float64
end

"""
    preObj_output(tempEl::Matrix{Float64}, vVolV::Vector{Float64}, vDist::Vector{Float64}, vAng::Vector{Float64}, vPearson::Float64)

Structure containing data related to the pre-object elongation process.

Fields:
* `tempEl`: Temporary data, likely a vector of values related to the ellipse.
* `vVolV`: Vector of calculated volumes of the pre-object.
* `vDist`: Vector of distances between ellipses in the elongation process.
* `vAng`: Vector of angles related to the ellipse orientations.
* `vPearson`: Pearson correlation coefficient, possibly related to ellipse properties.
"""
struct preObj_output
    tempEl::Matrix{Float64}
    vVolV::Vector{Float64}
    vDist::Vector{Float64}
    vAng::Vector{Float64}
    vPearson::Float64
end

"""
    TerminatorToOSCAR(Hdist::Vector{Int}, Hvect::Vector{Int}, Htotal::Vector{Int}, numbIn::Int, numbOut::Int, cutOut::Vector{Int}, lengthOut::Vector{Int})

Structure containing information about termination criteria and outlier analysis.

Fields:
* `Hdist`: Vector indicating outliers (negative values) and inliers (positive values) for distance data.
* `Hvect`: Vector indicating outliers (negative values) and inliers (positive values) for angle data.
* `Htotal`: Combined outlier information based on `Hdist` and `Hvect`.
* `numbIn`: Number of data points within termination criteria.
* `numbOut`: Number of data points outside termination criteria.
* `cutOut`: Indices of the start of outlier regions.
* `lengthOut`: Lengths of the outlier regions.
"""
struct TerminatorToOSCAR
    Hdist::Vector{Int}
    Hvect::Vector{Int}
    Htotal::Vector{Int}
    numbIn::Int
    numbOut::Int
    cutOut::Vector{Int}
    lengthOut::Vector{Int}
end

"""
    Objects3D(preok::Vector{Any}, infoDF::DataFrame, inis::Int, noInits::Matrix{Float64})

A structure containing information about 3D objects.

Fields:
* `preok`: Likely a vector or array of pre-objects, which might be intermediate data structures representing potential 3D objects.
* `infoDF`: Probably a DataFrame containing additional information about the objects, such as statistics, properties, or metadata.
* `inis`: An integer representing the number of initial objects or starting points.
* `noInits`: An array or vector containing information about objects that did not initiate or were discarded.
"""
struct Objects3D
    preok::Vector{Any}  # Assuming preok is a vector of vectors of floats
    infoDF::DataFrame            # Assuming infoDF is a DataFrame
    inis::Int                    # Assuming inis is an integer count
    noInits::Matrix{Float64}      # Assuming noInits is a matrix of floats
end

"""
    min_max_Zdef(folder::String; medZ=0)

This function calculates the minimum and maximum Z values (bounding box height) 
based on a user-provided median Z value (medZ) or automatically determines 
it from the data.

**Arguments:**

* `folder` (String): Path to the CSV file containing object data.
* `medZ` (Int, optional): User-provided median Z value. Default is 0.

**Returns:**

A `zParam` struct containing the calculated minimum Z, median Z, and maximum Z values.

**Functionality:**

1. Reads the CSV file using `CSV.read` and converts it to a DataFrame.
2. Defines indices for various columns in the DataFrame (e.g., indexX, indexZ, etc.).
3. Extracts data for bounding box width (`bbW`) and calculates the area (`areaW`) based on other columns (indexA and indexB).
4. If `medZ` is greater than 0 (user-provided):
    - Filters `bbW` to exclude values less than `medZ` and calculates the average (`maxZ`).
    - Filters `bbW` to exclude values greater than `medZ` and calculates the average (`minZ`).
5. If `medZ` is not provided:
    - Calculates the average area (`value1`) of all objects.
    - Filters `bbW` based on `value1` (objects with a larger area than the average).
    - Calculates the average (`medZ`) of the filtered `bbW`.
    - Filters `bbW` based on the calculated `medZ` (similar to the user-provided case).
6. Rounds the minimum, median, and maximum Z values to integers and calls a function `zParam` (not shown) with these values.

"""
function min_max_Zdef(folder::String; medZ=0)
    #
    #
    data = CSV.read(folder, delim='\t', DataFrame)
    #
    #
    indexX = 3
    indexY = 4
    indexArea = 2
    indexZ = 1
    indexAng = 5
    indexA = 6
    indexB = 7
    indexFeret = 8
    indexFeretX = 9
    indexFerety = 10
    indexFeretAngle = 11
    indexFeretmin = 12
    indexBX = 13
    indexBY = 14
    indexArea = 2
    indexBW = 15
    indexBH = 16
    abW = data[:, indexBW]
    bbW = data[:, indexBH]

    areaW = (data[:, indexA] ./ 2) .* (data[:, indexB] ./ 2) .* pi
    if medZ > 0
        var2 = [bbW[i] for i in 1:length(bbW) if medZ < bbW[i]]
        if length(var2) > 0
            maxZ = mean(var2)
        else
            maxZ = medZ
        end
        var2 = [bbW[i] for i in 1:length(bbW) if medZ > bbW[i]]
        if length(var2) > 0
            minZ = mean(var2)
        else
            minZ = medZ
        end
    else
        # indV = [i for i in 1:length(bbW) if percentile(bbW, 25)<bbW[i]<percentile(bbW,75)];
        # areaW = areaW[indV]; bbW = bbW[indV];
        # value1 = mean(areaW);
        # var2 = [bbW[i] for i in 1:length(areaW) if value1<areaW[i]]; medZ = mean(var2);
        value1 = mean(areaW)
        var2 = [bbW[i] for i in 1:length(areaW) if value1 < areaW[i]]
        value1 = mean(var2)
        medZ = (round(value1))
        # var2 = [bbW[i] for i in 1:length(bbW) if value1<bbW[i]]; medZ = mean(var2);
        # medZ = (round(medZ));
        var2 = [bbW[i] for i in 1:length(bbW) if medZ < bbW[i]]
        if isempty(var2) == false
            maxZ = mean(var2)
        else
            maxZ = medZ
        end
        # maxZ = mean(var2);
        var2 = [bbW[i] for i in 1:length(bbW) if medZ > bbW[i]]
        if isempty(var2) == false
            minZ = mean(var2)
        else
            minZ = medZ
        end
    end
    #
    minZ = Int(round(minZ))
    medZ = Int(round(medZ))
    maxZ = Int.(round(maxZ))
    return (zParam(minZ, medZ, maxZ))
end


"""
    data2d_chunking(file, xind, yind, areaind, zind, angind, aind, bind, typeind, nCh)

This function segments 2D data from a CSV file based on changes in a specified Z value (zind).

**Arguments:**

* `file` (String): Path to the CSV file containing data.
* `xind` (Int): Index of the X-coordinate column in the CSV file.
* `yind` (Int): Index of the Y-coordinate column in the CSV file.
* `areaind` (Int): Index of the area column in the CSV file.
* `zind` (Int): Index of the Z-coordinate column in the CSV file (used for segmentation).
* `angind` (Int): Index of the angle column in the CSV file.
* `aind` (Int): Index of the major axis length column in the CSV file.
* `bind` (Int): Index of the minor axis length column in the CSV file.
* `typeind` (Int): Index of the first custom data type column (potentially multiple columns).
* `nCh` (Int): Number of custom data type columns (following `typeind`).

**Returns:**

* `data2d` (Vector{Array{Float64}}): A vector of 2D object data slices, 
  where each slice represents a group of objects with the same Z value.
* `numberObj2D` (Vector{Int}): A vector containing the number of objects in each slice.

**Functionality:**

1. Reads the CSV file using `CSV.read` and converts it to a DataFrame (`data`).
2. Initializes empty vectors for storing 2D data slices (`data2d`) and the number of objects per slice (`numberObj2D`).
3. Initializes variables:
    * `Zi`: A temporary array to store data for a single 2D object slice (initialized to zeros).
    * `lookcount`: Counter for the number of objects in the current slice.
    * `zt`: Stores the current Z value for comparison.
    * `indexcount`: Counter for the slice index within `data2d`.
4. Iterates through each row (object) in `data`:
    * Checks if the current Z value (`data[i,zind]`) is different from the previous Z (`zt`).
        * If different:
            * Shorten `Zi` to remove unnecessary rows.
            * Append `Zi` to the `data2d` vector (represents a new 2D object slice).
            * Append the number of objects in `Zi` to the `numberObj2D` vector.
            * Reset `lookcount`, `zt`, and create a new `Zi` array.
    * Increments `lookcount` (number of objects in the current slice).
    * Extracts and stores data for the current object in `Zi`:
        * X-coordinate, Y-coordinate, area, angle, Z-coordinate, major axis length, minor axis length, tag (initialized to 0), 
          index within current slice (`lookcount`), index of the current slice (`indexcount`), and index within the original data (`i`).
    * For each custom data type column (up to `nCh`):
        * Extracts and stores the value from the corresponding column in `data` into `Zi`.
    * If it's the last row (i == size(data)[1]):
        * Shorten `Zi` to remove unnecessary rows.
        * Append `Zi` to the `data2d` vector.
5. Returns `data2d`, `data` (original data, potentially unnecessary), and `numberObj2D`.

"""
function data2d_chunking(file, xind, yind, areaind, zind, angind, aind, bind, typeind, nCh)
    # data = CSV.read(file, delim='\t', copycols=true)
    data = CSV.read(file, delim='\t', DataFrame)
    data2d = Vector{Array{Float64}}()
    numberObj2D = Vector{}()
    Zi = zeros(size(data)[1], 11 + nCh)
    lookcount = 0
    zt = data[1, zind]
    indexcount = 1
    for i in 1:size(data)[1]
        # println("i = ", i)
        if ((data[i, zind]) > zt)
            Zi = Zi[1:lookcount, :]
            push!(data2d, Zi)
            push!(numberObj2D, lookcount)
            # println("rest lookcount")
            lookcount = 0
            indexcount = indexcount + 1
            zt = data[i, zind]
            Zi = zeros(size(data)[1], 11 + nCh)
        end
        lookcount += 1
        Zi[lookcount, 1] = data[i, xind] #Xc
        Zi[lookcount, 2] = data[i, yind] #Yc
        Zi[lookcount, 3] = data[i, areaind] #Area
        Zi[lookcount, 4] = data[i, angind] #Angle
        Zi[lookcount, 5] = data[i, zind] #Z
        Zi[lookcount, 6] = data[i, aind] #Major exe
        Zi[lookcount, 7] = data[i, bind] #Minor exe
        Zi[lookcount, 8] = 0 #Tag of object
        Zi[lookcount, 9] = lookcount #Index in Zdata (index of the slice Z in data2d; i.e. if first 2d object is in slice 5, slice 5 will have index = 1)
        Zi[lookcount, 10] = indexcount #Index Z
        Zi[lookcount, 11] = i #Index in Data
        # println(size(data))
        for j in 1:nCh
            # println(typeind+j-1)
            # data[i,typeind+j-1]
            Zi[lookcount, 11+j] = data[i, typeind+j-1] #Index in Data
        end
        if (i == size(data)[1])
            Zi = Zi[1:lookcount, :]
            push!(data2d, Zi)
        end
        # if data[i,zind] == 160
        #     println("Meto esto ", Zi[lookcount,5])
        # end
    end
    # indexX = 1; indexY = 2; indexArea = 3; indexAng = 4; indexZ = 5; indexA = 6; indexB = 7; indexTag = 8; indexData = 9; indexInZ = 10; indexInRawData = 11;
    data2d, data, numberObj2D
end

"""
    sortPoints_ascending(xref,yref,xar,yar)

Sorts points based on their distance to a reference point (xref, yref).

**Arguments:**

* `xref`: X-coordinate of the reference point.
* `yref`: Y-coordinate of the reference point.
* `xar`: Vector of X-coordinates of points to be sorted.
* `yar`: Vector of Y-coordinates of points to be sorted.

**Returns:**
* `inMinSorted`: Indices of sorted points.
* `B`: Distances of sorted points to the reference point.
* `dist`: Original distances (not sorted).

**Functionality:**

1. Calculates the Euclidean distance between each point in `(xar, yar)` and the reference point `(xref, yref)`, storing the results in `dist`.
2. Sorts the points based on their distance to the reference point using `sortperm` and stores the indices in `inMinSorted`.
3. Returns the sorted indices and distances.
"""
function sortPoints_ascending(xref, yref, xar, yar)
    dist = zeros(length(xar))
    for i in 1:length(xar)
        dist[i] = sqrt((xar[i] .- xref) .^ 2 .+ (yar[i] .- yref) .^ 2)
        #dist[i] = ((xar[i].-xref).^2 .+ (yar[i].-yref).^2)
    end
    inMinSorted = sortperm(vec(dist))
    B = dist[inMinSorted]
    #dist = B
    return inMinSorted, B, dist
end

"""
    sortPoints_ascending_mapToZ(xref,yref,xar,yar,zs)

Sorts points based on their distance to a reference point (xref, yref) and maps them to their original Z values.

**Arguments:**

* `xref`: X-coordinate of the reference point.
* `yref`: Y-coordinate of the reference point.
* `xar`: Vector of X-coordinates of points to be sorted.
* `yar`: Vector of Y-coordinates of points to be sorted.
* `zs`: Vector of Z-values corresponding to the points.

**Returns:**
* `inMinSorted`: Indices of sorted points.
* `B`: Distances of sorted points to the reference point.
* `dist`: Original distances (not sorted).
* `Zord`: Z-values of sorted points.
* `indicesInEachZ`: Indices of points in each original Z slice, sorted by distance.

**Functionality:**

1. Calculates the Euclidean distance between each point in `(xar, yar)` and the reference point `(xref, yref)`, storing the results in `dist`.
2. Creates a vector `indicesInEachZ` to track the original indices of points within each Z slice.
3. Sorts the points based on their distance to the reference point using `sortperm` and stores the indices in `inMinSorted`.
4. Applies the sorting to `indicesInEachZ` and `dist` to maintain consistency.
5. Extracts the Z-values corresponding to the sorted points into `Zord`.
6. Returns the sorted indices, distances, original distances, sorted Z-values, and indices within each Z slice.
"""
function sortPoints_ascending_mapToZ(xref, yref, xar, yar, zs)
    dist = zeros(length(xar))
    for i in 1:length(xar)
        dist[i] = sqrt((xar[i] .- xref) .^ 2 .+ (yar[i] .- yref) .^ 2)
        #dist[i] = ((xar[i].-xref).^2 .+ (yar[i].-yref).^2)
    end
    indicesInEachZ = vcat(collect(1:findfirst(zs .!= zs[1])-1), collect(1:length(zs)-(findfirst(zs .!= zs[1])-1)))
    inMinSorted = sortperm(vec(dist))
    indicesInEachZ = indicesInEachZ[inMinSorted]
    B = dist[inMinSorted]
    Zord = Int.(zs[inMinSorted])
    #dist = B
    return inMinSorted, B, dist, Zord, indicesInEachZ
end

"""
    checkIntit(data2d_copy, i, j)

Determines if a given object (i, j) in `data2d_copy` is a new object or a merge of existing objects.

**Arguments:**

* `data2d_copy`: A 2D array containing object data.
* `i`: Index of the current slice in `data2d_copy`.
* `j`: Index of the object within the current slice.

**Returns:**

* `disappear_init`: A vector containing the X, Y, and Z coordinates of the object that merged with the current object (or zeros if no merge is detected).

**Functionality:**

1. **Initialization:** Defines indices for accessing different columns in `data2d_copy`.
2. **Calculate mean diameter:** Attempts to calculate a mean diameter but currently sets it to 0.
3. **Extract coordinates:** Extracts the X and Y coordinates of the current object.
4. **Check for previous slice:**
   * If not the first slice:
     * Concatenates data from the current and previous slices.
     * Sorts points based on distance to the current object using `sortPoints_ascending_mapToZ`.
     * Filters points based on object tag and distance.
     * If a nearby object is found (distance less than `mean_diameter`):
       * Marks the current object as a merge.
       * Returns the coordinates of the merged object.
5. **Check for current slice only (if first slice):**
   * Sorts points based on distance to the current object using `sortPoints_ascending`.
   * Filters points based on object tag and distance.
   * If a nearby object is found:
     * Increments the object tag of the current object.
     * Returns the coordinates of the merged object.
6. **Handles potential errors by returning a vector of zeros.**

**Key Points:**

* The function relies on `sortPoints_ascending` and `sortPoints_ascending_mapToZ` for distance calculations and sorting.
* The `mean_diameter` calculation is currently unused.
* The function modifies the `data2d_copy` array by incrementing the `indexTag` of merged objects.
* The returned `disappear_init` indicates the coordinates of the merged object.

**Potential Improvements:**

* Implement a meaningful calculation for `mean_diameter`.
* Consider using a more efficient data structure for storing object information.
* Explore alternative methods for detecting object mergers (e.g., bounding boxes, spatial indexing).
* Improve error handling and robustness.
"""
function checkIntit(data2d_copy, i, j)
    indexX = 1
    indexY = 2
    indexArea = 3
    indexAng = 4
    indexZ = 5
    indexA = 6
    indexB = 7
    indexType = 8
    indexTag = 8
    indexData = 9
    indexInZ = 10
    indexInRawData = 11
    # mean_diameter = sqrt(mean([mean(data2d_copy[i][:,indexArea]) for i in 1:length(data2d_copy)])/pi)*2
    # min_diameter = sqrt(mean([findmin(data2d_copy[i][:,indexArea])[1] for i in 1:length(data2d_copy)])/pi)*2
    # mean_diameter = sqrt(findmin(data2d_copy[i][:,indexArea])[1]/pi)*2
    # mean_diameter = sqrt(mean(data2d_copy[i][:,indexArea])/pi)
    mean_diameter = 0
    xc = data2d_copy[i][j, indexX]
    yc = data2d_copy[i][j, indexY]
    if i != 1
        xar = vcat(data2d_copy[i][:, indexX], data2d_copy[i-1][:, indexX])
        yar = vcat(data2d_copy[i][:, indexY], data2d_copy[i-1][:, indexY])
        zs = vcat(data2d_copy[i][:, indexInZ], data2d_copy[i-1][:, indexInZ])
        inMinSorted, B, dist, Zord, indicesInEachZ = sortPoints_ascending_mapToZ(xc, yc, xar, yar, zs)
        inMinSorted = [inMinSorted[k] for k in 1:length(inMinSorted) if data2d_copy[Zord[k]][indicesInEachZ[k], indexTag] == 1 && B[k] != 0]
        B = dist[inMinSorted]
        if length(B) != 0 && any(B .< mean_diameter)
            data2d_copy[i][j, indexTag] = 1
            disappear_init = hcat(data2d_copy[i][j, indexX], data2d_copy[i][j, indexY], data2d_copy[i][j, indexZ])
        end
    else
        xar = data2d_copy[i][:, indexX]
        yar = data2d_copy[i][:, indexY]
        inMinSorted, B, dist = sortPoints_ascending(xc, yc, xar, yar)
        # inMinSorted = inMinSorted[2:end]
        inMinSorted = [inMinSorted[k] for k in 1:length(inMinSorted) if data2d_copy[i][inMinSorted[k], indexTag] == 1 && B[k] != 0]
        B = dist[inMinSorted]
        if length(B) != 0 && any(B .< mean_diameter)
            data2d_copy[i][j, indexTag] += 1
            disappear_init = hcat(data2d_copy[i][j, indexX], data2d_copy[i][j, indexY], data2d_copy[i][j, indexZ])
        end
    end
    try
        return disappear_init
    catch
        disappear_init = zeros(1, 3)
        return disappear_init
    end
end

"""
    generate_points(xref, yref, angref, aref, bref)

Generates a set of points defining an ellipse centered at (xref, yref) with major axis length `aref`, minor axis length `bref`, and rotated by `angref` degrees counterclockwise.

**Arguments:**

* `xref`: X-coordinate of the ellipse center.
* `yref`: Y-coordinate of the ellipse center.
* `angref`: Rotation angle of the ellipse in degrees (counterclockwise).
* `aref`: Length of the major axis.
* `bref`: Length of the minor axis.

**Returns:**

* `xpins`: A vector of X-coordinates of the generated points.
* `ypins`: A vector of Y-coordinates of the generated points.

**Functionality:**

1. Converts the rotation angle `angref` from degrees to radians.
2. Creates a vector of angles `theta` from 0 to 2π with `nn` points (default 359).
3. Calculates the coordinates of points on an ellipse centered at the origin with major and minor axes aligned with the x and y axes.
4. Creates a rotation matrix based on the given angle `angref`.
5. Rotates the ellipse points using the rotation matrix.
6. Translates the rotated ellipse points to the center (xref, yref).
7. Returns the X and Y coordinates of the generated points.

**Note:**

* The function assumes a clockwise rotation matrix but converts the input angle `angref` to radians for consistency with the trigonometric functions.
* The number of points generated is determined by the `nn` parameter, which is set to 359 by default.
"""
function generate_points(xref, yref, angref, aref, bref)
    angref = (angref * pi) / 180
    nn = 359
    theta = range(0, 2 * pi, length=nn)
    coordEl = zeros(2, nn)
    coordEl[1, :] = aref .* cos.(theta)
    coordEl[2, :] = bref .* sin.(theta)

    rot_matrix = [cos(angref) -sin(angref); sin(angref) cos(angref)]
    coordN = rot_matrix * coordEl
    xpins = coordN[1, :] .+ xref
    ypins = coordN[2, :] .+ yref
    return (xpins, ypins)
end

"""
    OverlappedEllipses(aA, bA, xA, yA, angA, aB, bB, xB, yB, angB)

Determines the overlap between two ellipses.

**Arguments:**

* `aA`, `bA`: Major and minor axis lengths of ellipse A.
* `xA`, `yA`: Center coordinates of ellipse A.
* `angA`: Rotation angle of ellipse A in degrees.
* `aB`, `bB`: Major and minor axis lengths of ellipse B.
* `xB`, `yB`: Center coordinates of ellipse B.
* `angB`: Rotation angle of ellipse B in degrees.

**Returns:**
An `overlappedInfo` struct containing:
* `xdA`, `ydA`: X and Y coordinates of points on ellipse A.
* `xdB`, `ydB`: X and Y coordinates of points on ellipse B.
* `qA`: Number of points from ellipse A falling within ellipse B.
* `qB`: Number of points from ellipse B falling within ellipse A.

**Functionality:**

1. Generates points for ellipse A using `generate_points`.
2. Converts the rotation angle of ellipse B to radians.
3. Calculates the focal points of ellipse B.
4. Iterates over points of ellipse A:
   * Calculates distances from the point to the focal points of ellipse B.
   * If the sum of distances is less than or equal to 2 * aB, increments `qA`.
5. Generates points for ellipse B using `generate_points`.
6. Converts the rotation angle of ellipse A to radians.
7. Calculates the focal points of ellipse A.
8. Iterates over points of ellipse B:
   * Calculates distances from the point to the focal points of ellipse A.
   * If the sum of distances is less than or equal to 2 * aA, increments `qB`.
9. Returns an `overlappedInfo` struct containing the generated points and overlap counts.

**Note:**

* The function uses the focal point property of ellipses to determine overlap.
* The number of points generated for each ellipse affects the accuracy of the overlap calculation.
"""
function OverlappedEllipses(aA, bA, xA, yA, angA, aB, bB, xB, yB, angB)
    # A on B
    xdA, ydA = generate_points(xA, yA, angA, aA, bA)
    angB = (angB * pi) / 180
    cB = sqrt(abs((aB)^2 - (bB)^2))
    f_1_x_B = xB - cB * cos(angB)
    f_1_y_B = yB - cB * sin(angB)
    f_2_x_B = xB + cB * cos(angB)
    f_2_y_B = yB + cB * sin(angB)
    qA = 0
    for k in 1:length(xdA)
        dis1tB = sqrt((xdA[k] .- f_1_x_B) .^ 2 .+ (ydA[k] .- f_1_y_B) .^ 2)
        dis2tB = sqrt((xdA[k] .- f_2_x_B) .^ 2 .+ (ydA[k] .- f_2_y_B) .^ 2)
        if (dis1tB + dis2tB) <= 2 * aB
            qA = qA + 1
        end
    end
    # B on A
    xdB, ydB = generate_points(xB, yB, angB, aB, bB)
    angA = (angA * pi) / 180
    cA = sqrt(abs((aA)^2 - (bA)^2))
    f_1_x_A = xA - cA * cos(angA)
    f_1_y_A = yA - cA * sin(angA)
    f_2_x_A = xA + cA * cos(angA)
    f_2_y_A = yA + cA * sin(angA)
    qB = 0
    for k in 1:length(xdB)
        dis1tA = sqrt((xdB[k] .- f_1_x_A) .^ 2 .+ (ydB[k] .- f_1_y_A) .^ 2)
        dis2tA = sqrt((xdB[k] .- f_2_x_A) .^ 2 .+ (ydB[k] .- f_2_y_A) .^ 2)
        if (dis1tA + dis2tA) <= 2 * aA
            qB = qB + 1
        end
    end
    overlappedInfo(xdA, ydA, xdB, ydB, qA, qB)
end

"""
    LinearRegression(y, x::Array{Float64} = hcat(ones(length(y)),1:1:length(y)); points_to_fit::Int64 = length(y))

Performs linear regression on the given data.

**Arguments:**

* `y`: Dependent variable (response).
* `x`: Independent variables (predictors), default is a matrix with ones and indices.
* `points_to_fit`: Number of data points to use for fitting, default is all points.

**Returns:**
A `Line` struct containing:
* `beta`: Estimated regression coefficients.
* `ypredicted`: Predicted values of the dependent variable.
* `y`: Original dependent variable values.
* `supCI`: Upper confidence interval for the predicted values.
* `downCI`: Lower confidence interval for the predicted values.
* `Rsquared`: R-squared value.

**Functionality:**

1. Handles potential differences in the number of data points to fit (`points_to_fit`) compared to the total number of data points.
2. Calculates the regression coefficients using the least squares method.
3. Calculates the error terms and estimates the variance of the error.
4. Calculates confidence intervals for the predicted values.
5. Calculates the R-squared value as a measure of model fit.
6. Returns a `Line` struct containing the calculated values.

**Note:**

* The function assumes a linear relationship between the dependent and independent variables.
* The confidence intervals are based on a normal distribution assumption for the errors.
* The R-squared value is calculated as 1 minus the ratio of the sum of squared errors to the total sum of squares.
"""
function LinearRegression(y, x::Array{Float64}=hcat(ones(length(y)), 1:1:length(y)); points_to_fit::Int64=length(y))
    if points_to_fit != length(y)
        y1 = y[1:points_to_fit]
        y2 = y
        x1 = x[1:points_to_fit, :]
        x2 = x
    else
        y1 = y2 = y
        x1 = x2 = x
    end
    beta = inv(x1' * x1) * (x1' * y1)
    error = y1 - x1 * beta
    sigmaSquared = Statistics.var(error) * (length(y1) - 1) / (length(y1) - size(beta, 2))
    sigma = sqrt(sigmaSquared) * 1.86
    y2 = y
    ypredicted = x2 * beta
    supCI = ypredicted .+ sigma
    downCI = ypredicted .- sigma
    SSerr = sum((y2 .- ypredicted) .^ 2)
    SStot = sum((y2 .- mean(y2)) .^ 2)
    #MSE = SE/length(y2)
    # RMSE = sqrt(MSE)
    #rMSE = MSE/var(y2)
    #Rsquared = 1 - rMSE
    Rsquared = 1 - SSerr / SStot
    return Line(beta, ypredicted, y2, supCI, downCI, Rsquared)
end

"""
    Line_3D(;x::Array{Float64}=ones(0), y::Array{Float64}=ones(0), z::Array{Float64}=collect(range(1.0,length(x),step =1)), points_to_fit::Int64 = length(x))

Fits a 3D line to given X, Y, and Z coordinates.

**Arguments:**

* `x`: Array of X-coordinates.
* `y`: Array of Y-coordinates.
* `z`: Array of Z-coordinates (defaults to a linear sequence if not provided).
* `points_to_fit`: Number of points to use for the linear regression.

**Returns:**
A `Vector3D` struct containing:
* `xpred`: Predicted X-coordinates based on the fitted line.
* `ypred`: Predicted Y-coordinates based on the fitted line.
* `xcenter`, `ycenter`, `zcenter`: Center coordinates of the line segment.
* `xref`, `yref`: Coordinates of a reference point on the line (likely the end point).
* `betaX`, `betaY`: Coefficients of the linear regression for X and Y.
* `vx`, `vy`, `vz`: Components of the direction vector of the line.
* `vmodule`: Magnitude of the direction vector.

**Functionality:**

1. Creates a matrix `z` with ones as the first column and the original `z` values as the second column.
2. Performs linear regression for X and Y as functions of Z using `LinearRegression`.
3. Calculates predicted X and Y values (`xpred`, `ypred`).
4. Determines the center coordinates of the line segment.
5. Extracts regression coefficients for X and Y.
6. Calculates the reference point (xref, yref) as the end point of the predicted line.
7. Calculates the direction vector components (vx, vy, vz) and its magnitude (vmodule).
8. Returns a `Vector3D` struct containing the calculated values.

**Note:**

* The function assumes a linear relationship between X, Y, and Z.
* The `points_to_fit` parameter allows for fitting the line to a subset of the data.
* The function calculates various properties of the fitted line, including center, direction vector, and reference point.
"""
function Line_3D(; x::Array{Float64}=ones(0), y::Array{Float64}=ones(0), z::Array{Float64}=collect(range(1.0, length(x), step=1)), points_to_fit::Int64=length(x))
    #collect(range(1, stop=length(x), length=length(x)))
    #println(z);
    z = hcat(ones(length(z)), z)
    regX = LinearRegression(x, z, points_to_fit=points_to_fit)
    regY = LinearRegression(y, z, points_to_fit=points_to_fit)
    xpred = regX.ypredicted
    ypred = regY.ypredicted
    xcenter = (xpred[1] + xpred[end]) / 2
    ycenter = (ypred[1] + ypred[end]) / 2
    zcenter = (z[1, 2] + z[end, 2]) / 2
    betaX = regX.beta
    betaY = regY.beta
    # betaX = LinearRegression(x,z, points_to_fit=points_to_fit).beta #No haria falta hacerlo otra vez no?
    # betaY = LinearRegression(y,z, points_to_fit=points_to_fit).beta
    # xref = xpred[end] #[1 (z[end,2]+1)] * betaX
    # yref = ypred[end] #[1 (z[end,2]+1)] * betaY
    # # es lo mismo en matrizx que en lo otro, `pero el output te da como array y el Float64() solo acepta int
    # xref = xref[1]; yref = yref[1];
    xref = betaX[2] * z[end, 2] + 1 + betaX[1]
    yref = betaY[2] * z[end, 2] + 1 + betaY[1]
    vx = xpred[end] - xpred[1]
    vy = ypred[end] - ypred[1]
    vz = z[end, 2] - z[1, 2] + 1
    vmodule = sqrt(vx^2 + vy^2 + vz^2)
    return Vector3D(xpred, ypred, xcenter, ycenter, zcenter, xref, yref, betaX, betaY, vx, vy, vz, vmodule)
end

"""
    EllipsesConnector(data2d_copy, z, tempEl, nextEl_1, medSizeMin)

Connects two ellipses across consecutive slices of data.

**Arguments:**

* `data2d_copy`: A 2D array containing object data.
* `z`: Index of the current slice.
* `tempEl`: Data for the current ellipse.
* `nextEl_1`: Index of the current ellipse in the previous slice.
* `medSizeMin`: Minimum size for considering an ellipse as a potential connection.

**Returns:**
A `jointEllipses_output` struct containing:
* `nextEl_2`: Index of the connected ellipse in the next slice (or 0 if no connection).
* `nextStep`: Flag indicating if a connection was found (1) or not (0).
* `cx`: X-coordinate of the projected center of the current ellipse.
* `cy`: Y-coordinate of the projected center of the current ellipse.

**Functionality:**

1. **Initialization:**
   * Defines indices for accessing different columns in `data2d_copy`.
   * Initializes `nextStep` and `nextEl_2`.
   * Calculates `n_points` based on `medSizeMin`.
2. **Calculate ellipse properties:**
   * Extracts relevant data from `tempEl` based on `n_points` or the entire `tempEl` if `n_points` is larger than the data size.
   * Calculates the median angle, major axis, and minor axis of the extracted data.
3. **Calculate projected center:**
   * Fits a line to the extracted X and Y coordinates using `Line_3D` (if possible).
   * If line fitting fails, calculates the center as the median of X and Y coordinates.
4. **Calculate properties of the last ellipse in `tempEl`:**
   * Extracts the major axis, minor axis, angle, X, and Y coordinates of the last ellipse in `tempEl`.
5. **Find potential connections:**
   * Sorts points in the next slice based on distance to the projected center.
   * Iterates over the closest points:
     * Calculates overlap between the current ellipse and the potential connection.
     * If overlap is found, adds the point index to `in_sel`.
6. **Find the best connection:**
   * If potential connections were found:
     * Sorts the potential connections based on distance to the projected center.
     * Sets `nextStep` to 1 and `nextEl_2` to the index of the closest connection.
7. **Returns a `jointEllipses_output` struct containing connection information.**

**Note:**

* The `n_points` parameter is used to define a subset of data for calculations.
* The function calculates projected center coordinates and checks for overlapping ellipses in the next slice.
"""
function EllipsesConnector(data2d_copy, z, tempEl, nextEl_1, medSizeMin)
    indexX = 1
    indexY = 2
    indexArea = 3
    indexAng = 4
    indexZ = 5
    indexA = 6
    indexB = 7
    indexTag = 8
    indexData = 9
    indexInZ = 10
    indexInRawData = 11
    nextStep = 0
    nextEl_2 = nothing
    n_points = Int(round(medSizeMin))
    # projected cx and cy
    if size(tempEl, 1) >= n_points
        # xS = tempEl[end-n_points+1:end, indexX]
        # yS = tempEl[end-n_points+1:end, indexY]
        # zS = tempEl[end-n_points+1:end, indexZ]
        # angref = median(tempEl[end-n_points+1:end, indexAng]);
        # aref = median(tempEl[end-n_points+1:end, indexA])
        # bref = median(tempEl[end-n_points+1:end, indexB]);
        xS = tempEl[1:n_points, indexX]
        yS = tempEl[1:n_points, indexY]
        zS = tempEl[1:n_points, indexZ]
        angref = median(tempEl[1:n_points, indexAng])
        aref = median(tempEl[1:n_points, indexA])
        bref = median(tempEl[1:n_points, indexB])
    else
        xS = tempEl[:, indexX]
        yS = tempEl[:, indexY]
        zS = tempEl[:, indexZ]
        angref = median(tempEl[:, indexAng])
        aref = median(tempEl[:, indexA])
        bref = median(tempEl[:, indexB])
    end
    ztest = hcat(ones(length(xS)), collect(range(1.0, length(xS), step=1)))
    if det(ztest' * ztest) != 0
        # line = Line_3D(x = xS, y = yS, points_to_fit = n_points)
        line = Line_3D(x=xS, y=yS, z=zS)
        cx = line.xref
        cy = line.yref
    else
        cx = median(xS)
        cy = median(yS)
    end
    #
    aup = tempEl[end, indexA] ./ 2
    bup = tempEl[end, indexB] ./ 2
    angup = tempEl[end, indexAng] .* (-1)
    xup = tempEl[end, indexX]
    yup = tempEl[end, indexY]
    #
    indV, distV, dists = sortPoints_ascending(xup, yup, data2d_copy[z+1][:, indexX], data2d_copy[z+1][:, indexY])
    #
    k = 4
    if length(indV) < k
        k = length(indV)
    end
    in_sel = Vector{}()
    for i in 1:k
        at = (data2d_copy[z+1][indV[i], indexA] ./ 2)
        bt = (data2d_copy[z+1][indV[i], indexB] ./ 2)
        angt = data2d_copy[z+1][indV[i], indexAng] .* (-1)
        xt = (data2d_copy[z+1][indV[i], indexX])
        yt = (data2d_copy[z+1][indV[i], indexY])
        #
        shared = OverlappedEllipses(aup, bup, xup, yup, angup, at, bt, xt, yt, angt)
        percOv_pair = shared.AonB + shared.BonA
        #
        if percOv_pair > 0
            push!(in_sel, indV[i])
        else
            break
        end
    end
    #
    if length(in_sel) > 0
        indV2, distV2, dists2 = sortPoints_ascending(cx, cy, data2d_copy[z+1][in_sel, indexX], data2d_copy[z+1][in_sel, indexY])
        #
        nextStep = 1
        nextEl_2 = in_sel[indV2[1]]
        # at = (data2d_copy[z+1][nextEl_2,indexA]./2);
        # bt = (data2d_copy[z+1][nextEl_2,indexB]./2);
        # angt = data2d_copy[z+1][nextEl_2,indexAng].*(-1);
        # xt = (data2d_copy[z+1][nextEl_2,indexX]);
        # yt = (data2d_copy[z+1][nextEl_2,indexY]);
        # sharedL = OverlappedEllipses(aup, bup, xup, yup, angup, at, bt, xt, yt, angt)
        # percOv_pairL = sharedL.AonB + sharedL.BonA;
        # #
        # if percOv_pairL > 0
        #     nextStep = 1;
        # end
    end
    #
    jointEllipses_output(nextEl_2, nextStep, cx, cy)
end

"""
    info_Preobj_distZ(tempEl, index1=1, index2=size(tempEl, 1))

Calculates distances, angles, and Pearson correlation for a given set of points.

**Arguments:**
* `tempEl`: A matrix containing point data (X, Y, Z, etc.).
* `index1`: Starting index for the data subset (default: 1).
* `index2`: Ending index for the data subset (default: last row).

**Returns:**
An `infoPreobj` struct containing:
* `distV`: Vector of distances between consecutive points and the fitted line.
* `angV`: Vector of angles between consecutive vectors and a reference vector.
* `pearson_ind`: Pearson correlation coefficient between distances and angles.

**Functionality:**
1. Extracts relevant data from `tempEl` based on `index1` and `index2`.
2. Fits a line to the extracted data using `Line_3D`.
3. Calculates distances between points and the fitted line.
4. Calculates angles between consecutive vectors and a reference vector.
5. Calculates the Pearson correlation coefficient between distances and angles.
6. Returns an `infoPreobj` struct containing the calculated values.

**Note:**
* The function assumes a linear relationship between the X and Y coordinates and Z.
* The Pearson correlation coefficient is calculated for distances and angles.
* The function handles potential errors in line fitting by using a fallback method.
"""
function info_Preobj_distZ(tempEl, index1=1, index2=size(tempEl, 1))
    indexX = 1
    indexY = 2
    indexArea = 3
    indexAng = 4
    indexZ = 5
    indexA = 6
    indexB = 7
    indexTag = 8
    indexData = 9
    indexInZ = 10
    indexInRawData = 11
    #
    if index1 < 1 || index2 > size(tempEl, 1)
        xS = tempEl[:, indexX]
        yS = tempEl[:, indexY]
        zS = tempEl[:, indexZ]
        n_points = Int(length(xS))
    else
        xS = tempEl[index1:index2, indexX]
        yS = tempEl[index1:index2, indexY]
        zS = tempEl[index1:index2, indexZ]
        n_points = Int(length(xS))
    end
    ztest = hcat(ones(length(xS)), collect(range(1.0, length(xS), step=1)))
    if det(ztest' * ztest) != 0
        # line = Line_3D(x = xS, y = yS, points_to_fit = n_points)
        # println("zS = ", zS)
        line = Line_3D(x=xS, y=yS, z=zS)
        cx = line.xcenter
        cy = line.ycenter
        a = line.betaX
        pX = a[2] #slope
        b = line.betaY
        pY = b[2] #slope
        xpred = hcat(ones(length(tempEl[:, indexZ])), tempEl[:, indexZ]) * a
        ypred = hcat(ones(length(tempEl[:, indexZ])), tempEl[:, indexZ]) * b
    else
        cx = median(xS)
        cy = median(yS)
        pX = 0
        pY = 0
    end
    distV = Vector{Float64}()
    angV = Vector{Float64}()
    pearson_ind = 0
    if size(tempEl, 1) > 1
        #distance
        try
            distV = sqrt.((xpred .- tempEl[:, indexX]) .^ 2 .+ (ypred .- tempEl[:, indexY]) .^ 2)
        catch e
            println("No predicted line")
            indV_sorted, distV_sorted, distV = sortPoints_ascending(cx, cy, tempEl[:, indexX], tempEl[:, indexY])
        end
        #angle
        refV = [pX pY (tempEl[end, indexZ] - tempEl[1, indexZ])]
        for i in 1:size(tempEl, 1)-1
            dx = tempEl[i+1, indexX] - tempEl[i, indexX]
            dy = tempEl[i+1, indexY] - tempEl[i, indexY]
            testV = [dx dy 1]
            angt = cos(dot(testV, refV) / (norm(testV) * norm(refV)))
            push!(angV, angt)
        end
        #pearson
        # pearsonV = Vector{Float64}();
        if length(distV) > 1
            a = distV[2:end]
            b = angV
            pearson_ind = getCovariance(distV[2:end], angV)
        end
        # push!(pearsonV, pearson_ind)
    end
    infoPreobj(distV, angV, pearson_ind)
end

"""
    preObjElongation(sliceI, indexI, data2d, maxZinterval, minZinterval, medZinterval)

Performs object elongation analysis from a given starting point in a 2D data slice.

**Arguments:**
* `sliceI`: Index of the starting slice in `data2d`.
* `indexI`: Index of the starting object within the slice.
* `data2d`: 2D array containing object data.
* `maxZinterval`: Maximum number of slices to consider for elongation.
* `minZinterval`: Minimum number of slices for considering a connection.
* `medZinterval`: Size of the data subset for calculating distances and angles.

**Returns:**
A `preObj_output` struct containing:
* `tempEl`: Elongated object data.
* `vVolV`: Vector of volumes for each slice of the elongated object.
* `vDist`: Vector of distances between consecutive points and the fitted line.
* `vAng`: Vector of angles between consecutive vectors and a reference vector.
* `vPearson`: Pearson correlation between distances and angles.

**Functionality:**
1. Creates a copy of the input data `data2d`.
2. Initializes variables for tracking object elongation and calculations.
3. Sets the starting point for the elongation process.
4. Calculates the volume of the initial object and adds it to `vVolV`.
5. Iterates through subsequent slices:
   * Uses `EllipsesConnector` to find potential connections between the current object and objects in the next slice.
   * If a connection is found:
     * Updates the current object with data from the connected object.
     * Calculates the volume of the elongated object and adds it to `vVolV`.
   * If no connection is found, stops the elongation process.
6. Calculates distances, angles, and Pearson correlation using `info_Preobj_distZ`.
7. Returns a `preObj_output` struct containing the elongated object data and calculated values.

**Note:**
* The function relies on `EllipsesConnector` and `info_Preobj_distZ` for core functionalities.
* The `tempEl` variable stores the data for the elongated object.
* The `vVolV` vector tracks the volume of the object at each step of the elongation process.
* The function calculates distances, angles, and Pearson correlation for the final elongated object.
"""
function preObjElongation(sliceI, indexI, data2d, maxZinterval, minZinterval, medZinterval)
    data2d_copy = deepcopy(data2d)
    indexX = 1
    indexY = 2
    indexArea = 3
    indexAng = 4
    indexZ = 5
    indexA = 6
    indexB = 7
    indexType = 8
    indexTag = 8
    indexData = 9
    indexInZ = 10
    indexInRawData = 11
    # println("a ", data2d_copy[sliceI][indexI,12])
    # println("b ", data2d_copy[sliceI][indexI,13])
    # outputs def.
    vVolV = Vector{Float64}()
    # vDist = Vector{Float64}(); vAng = Vector{Float64}()
    # vPearson = nothing;
    # Initiator
    nextEl_1 = indexI
    z = sliceI
    tempEl = data2d_copy[z][nextEl_1, :]'
    pearson_ind = 0
    volPre = tempEl[1, indexA] .* tempEl[1, indexB] .* pi
    push!(vVolV, volPre)
    #elongation
    while z < sliceI + maxZinterval - 1 && z < size(data2d_copy, 1)
        ellipseInfo = EllipsesConnector(data2d_copy, z, tempEl, nextEl_1, minZinterval)
        goQ = ellipseInfo.nextStep
        nextEl_2 = 0
        if goQ == 1
            nextEl_2 = ellipseInfo.nextEl_2
            #
            # println("nextEl_2 = ", nextEl_2)
            # println("previous Z = ", z)
            nextEl_1 = nextEl_2
            z = z + 1
            # println("nextEl_1 = ", nextEl_1)    #ojito que ya no es z+1
            # println("next Z = ", z)
            # println("que cojones pone en la línea? = ", data2d_copy[z][nextEl_1,:]')
            tempEl = vcat(tempEl, data2d_copy[z][nextEl_1, :]')
            #
            volPre = sum(tempEl[:, indexA] .* tempEl[:, indexB] .* pi)
            push!(vVolV, volPre)
            #
        else
            break
        end
    end
    #
    info = info_Preobj_distZ(tempEl, 1, medZinterval)
    distV = info.dist
    angV = info.ang
    pearson = info.pearson
    vDist = distV
    vAng = angV
    vPearson = pearson
    # println("a' ", tempEl[:,12])
    # println("b' ", tempEl[:,13])
    #
    preObj_output(tempEl, vVolV, vDist, vAng, vPearson)
end

"""
    outliersDetection(vDistV, sliceVector, medSizeMin)

Detects outliers in a given data vector `vDistV` using a linear regression-based approach.

**Arguments:**
* `vDistV`: The data vector to analyze.
* `sliceVector`: A vector of indices, likely used for indexing purposes.
* `medSizeMin`: The minimum size of the data subset used for initial regression.

**Returns:**
* `H`: A vector indicating outliers (negative values) and inliers (positive values).
* `y1`: A filtered version of `vDistV` excluding outliers.
* `cutPoints`: Indices of the start of outlier regions.
* `returnPoints`: Indices of the end of outlier regions.

**Functionality:**
1. Fits a linear regression model to the initial part of `vDistV` to estimate a trend.
2. Calculates upper and lower confidence intervals based on the model and a robust estimate of the standard deviation (interquartile range).
3. Identifies outliers as points outside the confidence intervals and marks them in the `H` vector.
4. Filters the original data to exclude outliers and stores the result in `y1`.
5. Detects contiguous outlier regions and stores their starting and ending indices in `cutPoints` and `returnPoints`.

**Note:**
* The function uses a linear regression model to establish a baseline and identify deviations.
* Outliers are defined as points outside a specified confidence interval.
* The use of `sliceVector` is not explicitly defined in the provided code and might require further context.
"""
function outliersDetection(vDistV, sliceVector, medSizeMin)
    #ztest
    a = vDistV[1:medSizeMin]
    ztest = hcat(ones(length(vDistV[1:medSizeMin])), range(1, length(vDistV[1:medSizeMin]), step=1))
    if det(ztest' * ztest) != 0
        regD = LinearRegression(vDistV, points_to_fit=medSizeMin)
        ypredicted = regD.ypredicted
        y = regD.y
        supCI = regD.supCI
        # downCI = regD.downCI; Rsquared = regD.Rsquared;
        # sigma = findmax(y[1:medSizeMin])[2][1] - findmin(y[1:medSizeMin])[2][1]
        sigma = abs(percentile(y[1:medSizeMin], 25) - percentile(y[1:medSizeMin], 75))
        # sigma = percentile(y[1:medSizeMin], 0) - percentile(y[1:medSizeMin], 100);
        supCI = ypredicted .+ sigma
        downCI = ypredicted .- sigma
    else
        y = vDistV
        # sigma = findmax(y[1:medSizeMin])[2][1] - findmin(y[1:medSizeMin])[2][1]
        sigma = percentile(y[1:medSizeMin], 25) - percentile(y[1:medSizeMin], 75)
        ypredicted = ones(length(y), 1) .* median(y[1:medSizeMin])
        supCI = ypredicted .+ sigma
        downCI = ypredicted .- sigma
    end
    #
    H = sliceVector
    y1 = Vector{Float64}()
    for point in 1:length(y)
        #b = findmin(abs.(ypredicted.-y[point]))[2]
        b = point
        if y[point] > supCI[b] || y[point] < downCI[b]
            H[point] = H[point] .* -1
        else
            push!(y1, vDistV[point])
        end
    end
    cutPoints = Vector{Int64}()
    returnPoints = Vector{Int64}()
    for i in 1:length(H)
        if H[i] < 0
            push!(cutPoints, i)
            c = 1
            for j in i:length(H)-1
                if H[j+1] > 0
                    c = 0
                    push!(returnPoints, j)
                    break
                end
            end
            if c == 1
                push!(returnPoints, length(H))
            end
        end
    end
    H, y1, cutPoints, returnPoints
end

"""
    TerminatorReturns(vDistV, angPreok, n_points)

Determines termination criteria based on distance and angle data.

**Arguments:**
* `vDistV`: Vector of distance values.
* `angPreok`: Vector of angle values.
* `n_points`: A threshold value.

**Returns:**
A `TerminatorToOSCAR` struct containing termination criteria information.

**Functionality:**
1. Performs outlier detection on both distance and angle data using `outliersDetection`.
2. Converts outlier information into binary values (1 for outlier, -1 for inlier).
3. Calculates a combined outlier metric (`HT`).
4. Counts the number of points within and outside the termination criteria.
5. Identifies cut points and slice lengths based on the termination criteria.
6. Creates a `TerminatorToOSCAR` struct and returns it.

**Note:**
* The function relies on the `outliersDetection` function to identify outliers.
"""
function TerminatorReturns(vDistV, angPreok, n_points)
    # println("size of d ", size(vDistV))
    # println("size of v ", size(angPreok))
    # println("n_points ", n_points)
    ##PROBANDO LAS FUNCIONES DE ARRIBA SON LAS BUENA outliersDetection
    HD, yD, cutPointsD, returnPointsD = outliersDetection(vDistV, collect(1:1:length(vDistV)), n_points)
    HV, yV, cutPointsV, returnPointsV = outliersDetection(angPreok, collect(1:1:length(angPreok)), n_points)
    # HD, yD, cutPointsD, returnPointsD = outliers_from_zero(info.dist[2:end])
    # HV, yV, cutPointsV, returnPointsV = outliers_from_zero(info.ang)
    # println("size of Hd ", size(HD))
    # println("size of Hv ", size(HV))
    HD = Int.(((HD ./ abs.(HD)) .- 1) ./ (-2))
    HV = Int.(((HV ./ abs.(HV)) .- 1) ./ (-2))
    HT = (HD .+ HV)
    #
    cin = 0
    cout = 0
    cutPointsf = Vector{Float64}()
    for i in 1:length(HT)
        if HT[i] > 1
            if i <= n_points
                cin = cin + 1
            else
                cout = cout + 1
                push!(cutPointsf, i)
            end
        end
    end
    lenghtSlicesf = diff(cutPointsf) .- 1
    #
    TerminatorToOSCAR(HD, HV, HT, cin, cout, cutPointsf, lenghtSlicesf)
end

"""

    getCovariance(var1, var2)

Calculates the Pearson correlation coefficient between two vectors.

**Arguments:**
* `var1`: First vector of data.
* `var2`: Second vector of data.

**Returns:**
* `pearson_ind`: Pearson correlation coefficient.

**Functionality:**
1. Checks if the standard deviations of `var1` and `var2` are zero.
2. If both standard deviations are zero, sets the Pearson correlation coefficient to 1.
3. Otherwise, calculates the covariance between `var1` and `var2` and divides it by the product of their standard deviations to obtain the Pearson correlation coefficient.

**Note:**
* The function handles cases where the standard deviation of one or both vectors is zero.
* The Pearson correlation coefficient measures the linear relationship between two variables.
"""
function getCovariance(var1, var2)
    pearson_ind = nothing
    if (std(var1) * std(var2)) == 0
        pearson_ind = 1
    else
        pearson_ind = cov(var1, var2) / (std(var1) * std(var2))
    end
    pearson_ind
end

"""
    summary_objs(RefStruct, nCh = 1) -> DataFrame

Summarizes object properties from a reference structure containing segmented objects.

# Arguments
- `RefStruct`: A structure containing object data, expected to have a `preok` field with segmented object properties.
- `nCh::Int`: Number of additional data channels to include in the summary.

# Returns
- `DataFrame`: A table summarizing object properties including position, orientation, and size.

# Details
1. Extracts object properties such as center coordinates, principal axes, angles, and volumes.
2. Computes median values for key parameters.
3. Uses linear regression to determine 3D orientation when possible.
4. Constructs a structured `DataFrame` with relevant properties for all objects.

# Examples
```julia
using DataFrames

objects_summary = summary_objs(myRefStruct, nCh = 2)
```
"""
function summary_objs(RefStruct, nCh = 1)
    indexX = 1; indexY = 2; indexArea = 3; indexAng = 4; indexZ = 5; indexA = 6; indexB = 7; indexTag = 8; indexData = 9; indexInZ = 10; indexInRawData = 11;
    # indexType = 12;
    objects = RefStruct.preok;
    matrixData = zeros(size(objects,1), indexInRawData+1+nCh); c = 0;
    for i in 1:size(objects,1)
        x = objects[i][:,indexX]
        y = objects[i][:,indexY]
        z = objects[i][:,indexZ]
        angle = objects[i][:,indexAng]
        a = objects[i][:,indexA]
        b = objects[i][:,indexB]
        area = objects[i][:,indexArea]
        ztest = hcat(ones(length(x)), collect(range(1.0,length(x),step =1)));
        if det(ztest'*ztest) != 0
            vector3D = Line_3D(x = x, y = y, z = z)
            cx = vector3D.xcenter; cy = vector3D.ycenter;
            cz = vector3D.zcenter;
            vx = vector3D.vx; vy = vector3D.vy; vz = vector3D.vz;
        else
            cx = median(x); cy = median(y);
            cz = median(z);
            vx = 0.0; vy = 0.0; vz = 0.0;
        end
        medianA = median(a)
        medianB = median(b)
        medianC = medianC = sqrt(vx^2+vy^2+vz^2)
   
        medianAngleXY = median(angle)
        vol = sum(area);
        label = objects[i][1,indexInRawData]
        dataV = hcat(cx, cy, cz, medianAngleXY, medianA, medianB, medianC, vol, vx, vy, vz, label)
        for j in 1:nCh
            tempt = (objects[i][:,indexInRawData+j])
            dataV = hcat(dataV, sum(tempt));
        end
       
        c = c + 1; 
        matrixData[c, :] = dataV;
    end
    names = vcat([:Xcenter, :Ycenter, :Zcenter, :AngleXY, :A, :B, :C, :Vol, :Vx, :Vy, :Vz, :Label], Symbol.(:Channel,axes(ones(nCh),1)))
    objects3D_df = DataFrame(Tables.table(matrixData, header = names))

    return(objects3D_df)
end

"""
    IJDefault(data)::Int

Computes an iterative threshold value for a given `data` array. This function searches for the first and last nonzero elements in the data, then iteratively calculates a threshold by dividing the data into two weighted sums.

It returns an integer representing the computed threshold.

# Examples
```julia
data = [0, 0, 0, 5, 10, 15, 10, 5, 0, 0]
threshold = IJDefault(data)
println(threshold)  # Example output: 5
```
"""
function IJDefault(data)
    maxValue = length(data) - 1
    min_idx = 1
    
    while data[min_idx] == 0 && min_idx < maxValue
        min_idx += 1
    end
    
    max_idx = maxValue
    while data[max_idx] == 0 && max_idx > 1
        max_idx -= 1
    end
    
    if min_idx >= max_idx
        return div(length(data), 2)
    end
    
    movingIndex = min_idx
    inc = max(div(max_idx, 40), 1)
    result = 0.0
    
    while true
        sum1 = sum2 = sum3 = sum4 = 0.0
        
        for i in min_idx:movingIndex
            sum1 += i * data[i]
            sum2 += data[i]
        end
        
        for i in (movingIndex+1):max_idx
            sum3 += i * data[i]
            sum4 += data[i]
        end
        
        result = (sum1 / sum2 + sum3 / sum4) / 2.0
        movingIndex += 1
        
        if movingIndex + 1 > result || movingIndex >= max_idx - 1
            break
        end
    end
    
    return round(Int, result)
end

"""
    Threshold(img::AbstractArray; noWhite::Bool, noBlack::Bool, doIwhite::Bool, doIset::Bool, doIlog::Bool, doIstackHistogram::Bool)::Int

Computes a threshold value for an image using histogram-based analysis. The function processes both 8-bit and 16-bit grayscale images and optionally excludes black and white pixels from the histogram calculation.

If `doIstackHistogram` is `true`, histograms from image stacks are merged. The computed threshold is obtained using `IJDefault`.

# Arguments
- `img::AbstractArray`: Input grayscale image or image stack.
- `noWhite::Bool`: If `true`, ignores the maximum intensity (white) in the histogram.
- `noBlack::Bool`: If `true`, ignores the minimum intensity (black) in the histogram.
- `doIwhite::Bool`: (Unused in this function but reserved for further operations.)
- `doIset::Bool`: (Unused in this function but reserved for further operations.)
- `doIlog::Bool`: If `true`, prints the computed threshold value.
- `doIstackHistogram::Bool`: If `true`, merges histograms when processing image stacks.

# Returns
- `Int`: The computed threshold value.

# Examples
```julia
using Images

img = rand(Gray{N0f8}, 100, 100)  # Generate a random grayscale image
threshold = Threshold(img; noWhite=true, noBlack=false, doIwhite=false, doIset=false, doIlog=true, doIstackHistogram=false)
println("Computed threshold: ", threshold)
```
"""
function Threshold(img::AbstractArray; noWhite::Bool, noBlack::Bool, doIwhite::Bool, doIset::Bool, doIlog::Bool, doIstackHistogram::Bool)
    if img === nothing
        return nothing
    end

    bitDepth = eltype(img) <: Gray{N0f8} ? 8 : 16  # Detect if 8-bit or 16-bit
    max_intensity = bitDepth == 8 ? 255 : 65535

    # Convert to integer matrix
    img_matrix = Int.(round.(Gray.(img) .* max_intensity))

    # If the image is a stack, merge histograms
    if ndims(img_matrix) == 3 && doIstackHistogram
        img_stack = [img_matrix[:, :, i] for i in 1:size(img_matrix, 3)]
    else
        img_stack = [img_matrix]
    end

    # Compute histogram
    data = zeros(Int, max_intensity + 1)
    for slice in img_stack
        for val in slice
            data[val + 1] += 1
        end
    end

    if noBlack
        data[1] = 0
    end
    if noWhite
        data[end] = 0
    end

    # Find histogram range
    minbin = findfirst(x -> x > 0, data)
    maxbin = findlast(x -> x > 0, data)

    if isnothing(minbin) || isnothing(maxbin) || minbin >= maxbin
        return 255
    end

    data2 = data[minbin:maxbin]

    # Apply thresholding
    threshold = length(data2) < 2 ? 0 : IJDefault(data2)
    threshold += minbin - 1  # Adjust for offset

    # Log threshold
    if doIlog
        println("Threshold: ", threshold)
    end

    return threshold
end

"""
    getminR(img::AbstractArray)::Float64

Computes the minimum representative radius (`minR`) of an object in a 3D grayscale image stack. The function thresholds the central slice of the stack, binarizes it, applies a distance transform, and determines the mean distance of nonzero values.

# Arguments
- `img::AbstractArray`: A 3D grayscale image stack.

# Returns
- `Float64`: The estimated minimum radius.

# Details
1. The function selects the middle slice of the image stack and computes a threshold using `Threshold`, excluding black and white pixels.
2. The selected slice is binarized using the computed threshold.
3. A distance transform is applied to the binarized image to measure the Euclidean distance to the nearest background pixel.
4. The mean of all nonzero distances is computed, divided by two, and incremented by one to obtain `minR`.

# Examples
```julia
using Images

img = rand(Gray{N0f8}, 100, 100, 10)  # Generate a random 3D grayscale image stack
minR = getminR(img)
println("Computed minR: ", minR)
```
"""
function getminR(img::AbstractArray)
    threshold = Threshold(img[:,:,round(Int,size(img,3)/2)],noWhite = true, noBlack = true,doIwhite = false, doIset= false,doIlog=false,doIstackHistogram = false)
    bitDepth = eltype(img) <: Gray{N0f8} ? 8 : 16  # Detect if 8-bit or 16-bit
    max_intensity = bitDepth == 8 ? 255 : 65535
    # Convert to integer matrix
    img_matrix = Int.(round.(Gray.(img) .* max_intensity))

    binaryImage = binarizeImage(Float64.(img_matrix[:,:,round(Int,size(img,3)/2)]),Float64(threshold))
    distances::Matrix{Float64} = distance_transform(feature_transform(Bool.(binaryImage) .== false))
    distances[findall(x -> x == 0.0, distances)] .= typemin(Float64)
    minR = round(mean(distances[findall(distances .!= -Inf)]))/2+1
    return minR
end


"""
    get_bounding_box(img::AbstractArray; minSize::Int=0) -> Union{BoundingBox, Nothing}

Computes the bounding box of the nonzero elements in a 2D grayscale or binary image. If a `minSize` is specified, the bounding box is symmetrically expanded (where possible) to ensure it meets or exceeds that size along both axes.

# Arguments
- `img::AbstractArray`: A 2D array representing a grayscale or binary image.
- `minSize::Int=0`: Optional. Minimum width and height of the bounding box. If the actual size is smaller, the bounding box is expanded symmetrically within image bounds.

# Returns
- `BoundingBox`: A structure with the fields `(x_max, x_min, y_max, y_min)`, representing the coordinates of the bounding box.
- `nothing`: Returned if the image contains no nonzero elements.

# Bounding Box Fields
- `x_max`: Maximum row (vertical) index of the nonzero pixels.
- `x_min`: Minimum row (vertical) index of the nonzero pixels.
- `y_max`: Maximum column (horizontal) index of the nonzero pixels.
- `y_min`: Minimum column (horizontal) index of the nonzero pixels.

# Method
1. Finds all nonzero pixel coordinates using `findall`.
2. Extracts the x (row) and y (column) indices.
3. Calculates min and max for each axis.
4. Optionally expands the bounding box to ensure a minimum size.

# Example
```julia
using Images

img = zeros(Float64, 100, 100)
img[30:70, 40:80] .= 1.0  # Create a rectangular object

bbox = get_bounding_box(img, minSize=50)
println("Bounding Box: ", bbox)
```
""" 
function get_bounding_box(img::AbstractArray, minSize::Int = 0)
    coords_ = findall(x -> x != 0.0, img)
    if isempty(coords_)
        return nothing
    end

    xCoords = [x[1] for x in coords_]
    yCoords = [x[2] for x in coords_]

    xMin = minimum(xCoords)
    xMax = maximum(xCoords)
    yMin = minimum(yCoords)
    yMax = maximum(yCoords)

    img_height = size(img, 1)
    img_width = size(img, 2)

    # Adjust X bounds if needed
    if (xMax - xMin + 1) < minSize
        needed = minSize - (xMax - xMin + 1)
        addX = ceil(Int, needed / 2)
        xMin = max(1, xMin - addX)
        xMax = min(img_height, xMax + addX)
        # Ensure size is exactly minSize if possible
        if (xMax - xMin + 1) < minSize
            if xMin > 1
                xMin = max(1, xMin - (minSize - (xMax - xMin + 1)))
            elseif xMax < img_height
                xMax = min(img_height, xMax + (minSize - (xMax - xMin + 1)))
            end
        end
    end

    # Adjust Y bounds if needed
    if (yMax - yMin + 1) < minSize
        needed = minSize - (yMax - yMin + 1)
        addY = ceil(Int, needed / 2)
        yMin = max(1, yMin - addY)
        yMax = min(img_width, yMax + addY)
        # Ensure size is exactly minSize if possible
        if (yMax - yMin + 1) < minSize
            if yMin > 1
                yMin = max(1, yMin - (minSize - (yMax - yMin + 1)))
            elseif yMax < img_width
                yMax = min(img_width, yMax + (minSize - (yMax - yMin + 1)))
            end
        end
    end

    return BoundingBox(xMax, xMin, yMax, yMin)
end

"""
    custom_median_filter2D(img::Array{<:Number,2}, kernel::BitArray{2}) -> Matrix{eltype(img)}

Applies a custom 2D median filter to an image using a binary kernel that defines the neighborhood shape. Unlike standard square-shaped median filters, this function allows arbitrary structuring elements (e.g. circular, elliptical) provided as binary masks.

# Arguments
- `img::Array{<:Number,2}`: A 2D grayscale image (e.g., `Matrix{Float64}`).
- `kernel::BitArray{2}`: A binary matrix where `true` values define the shape of the filtering neighborhood. Must have odd dimensions to ensure a central pixel.

# Returns
- `Matrix{eltype(img)}`: The filtered image where each pixel is the median of its defined neighborhood.

# Details
1. Pads the input image with zeros based on the kernel size.
2. Computes relative offsets from the kernel center for all `true` positions in the kernel.
3. For each pixel in the image:
   - Gathers neighborhood values using the offsets.
   - Computes the median of the neighborhood.
   - Assigns it to the corresponding pixel in the output image.

# Example
```julia
img = rand(100, 100)
kernel = trues(5, 5)              # 5×5 square kernel
filtered = custom_median_filter2D(img, kernel)

```
# Notes
Padding ensures boundary handling by filling with zeros.

Kernel must be small enough to avoid indexing beyond the padded image.

Useful for denoising while preserving edges, especially with shaped kernels.
"""
function custom_median_filter2D(img::Array{<:Number,2}, kernel::BitArray{2})
    sx,sy = size(kernel)
    ox, oy = div.((sx, sy), 2)
    center = CartesianIndex(ox+1, oy+1)
    padded = padarray(img, Fill(0, (ox, oy)))
    output = similar(img)

    inds = CartesianIndices(kernel)
    offsets = [I .- center for I in inds if kernel[I]]

    for I in CartesianIndices(img)
        values = [padded[I .+ δ] for δ in offsets]
        output[I] = median(values)
    end
    return output
end

"""
    nuclei_processing(img::AbstractArray; kernel=nothing) -> Union{Matrix{Float64}, Array{Float64, 3}}

Applies a nuclei enhancement filter to a 2D or 3D image stack. The method uses median filtering, Gaussian smoothing, and morphological opening to emphasize bright nuclei-like structures while suppressing the background. If `kernel` is not provided, the filtering kernel size is inferred automatically.

# Arguments
- `img::AbstractArray`: A 2D or 3D grayscale image (matrix or volume).
- `kernel`: Optional. An integer value that sets the size of the structuring elements. If `nothing`, an automatic estimate is computed using `getminR`.

# Returns
- `Matrix{Float64}` or `Array{Float64, 3}`: A filtered image or image stack highlighting nuclei-like features. The result is scaled to the [0, 1] range with rounded precision.

# Method
1. Determines kernel size (`minR`) based on the image if not provided.
2. Constructs:
   - A small median filter kernel (`minR`)
   - A larger opening kernel (`2 * minR`)
   - A Gaussian kernel (`2 * minR`)
3. Iterates over each z-slice (in case of 3D images):
   - Extracts a bounding box around nonzero regions using `get_bounding_box`.
   - Applies 2D median filtering to the ROI.
   - Smooths the result with Gaussian filtering.
   - Applies morphological opening.
   - Subtracts the smoothed version from the median-filtered ROI to enhance signal.
4. Assembles results into a new image stack and rescales output to [0, 1] in 8-bit-like precision.

# Example
```julia
img = load("nuclei_stack.tif")  # 3D stack
processed = nuclei_processing_faster(img)
save("enhanced_stack.tif", processed)
```
"""
function nuclei_processing(img::AbstractArray; kernel = nothing)::Tuple{Union{Matrix{Float64}, Array{Float64, 3}},Float64}
    result_img::Array{Float64} = zeros(Float64, size(img))
    if kernel == nothing
        minR = getminR(img)
    else
        minR = kernel
    end
    println("MinR = ",minR)
    # kernel1 = Bool.(drawKernel(minR)[1][:,:,1:1])
    # kernel2 = Bool.(drawKernel(minR*2)[1][:,:,1:1])
    # k_gauss = Kernel.gaussian(minR*2)[:,:,1:1]
    # img = custom_median_filter(img,kernel1)
    # img_copy = imfilter(img, k_gauss)
    # img_copy = opening(img_copy,kernel2)
    # result_img = img .- img_copy
    # result_img = round.(result_img.*255)./255
    # return result_img
    kernel1 = Bool.(drawKernel(minR)[1])
    kernel2 = Bool.(drawKernel(minR*2)[1])
    k_gauss = Kernel.gaussian(minR*2)
    Threads.@threads for i = tqdm(1:size(img,3))
    # for i = 1:size(img,3)
        boundingBox = get_bounding_box(img[:,:,i])
        if typeof(boundingBox) == BoundingBox
            img_median = custom_median_filter2D(img[boundingBox.minX:boundingBox.maxX,boundingBox.minY:boundingBox.maxY,i],kernel1)
            img_copy = imfilter(img_median, k_gauss)
            img_copy = opening(img_copy,kernel2)
            result_img[boundingBox.minX:boundingBox.maxX,boundingBox.minY:boundingBox.maxY,i] = img_median .- img_copy
        end
    end
    result_img = round.(result_img.*255)./255
    return result_img, minR
end


"""
ObjSplitter3D(file, medZ, minZ=0.5*medZ, maxZ=1.5*medZ, nCh=1)

Splits 3D data into objects based on specified criteria.

**Arguments:**
* `file`: Path to the input data file.
* `medZ`: Target median Z value for object segmentation.
* `minZ`, `maxZ`: Minimum and maximum Z values for object consideration (optional).
* `nCh`: Number of additional channels in the data.

**Returns:**
* `Objects3D`: A struct containing information about the detected objects.
* `counter_cin`: A counter for a specific condition (not fully explained).
* `count_pearson`: A counter for a specific condition (not fully explained).

**Functionality:**
1. Reads the input data from the file and creates a 2D representation.
2. Initializes variables for object tracking and statistics.
3. Iterates over each slice of the 2D data:
   * For each object in the slice:
     * Checks if the object is a new initiator or a continuation of an existing object.
     * If a new initiator:
       * Performs object elongation using `preObjElongation`.
       * Applies termination criteria using `TerminatorReturns`.
       * Checks for potential object mergers and updates object data accordingly.
     * Updates object counters and statistics.
4. Creates a DataFrame summarizing object information.
5. Returns an `Objects3D` struct, `counter_cin`, and `count_pearson`.

**Note:**
* The function relies on several other functions: `data2d_chunking`, `checkIntit`, `preObjElongation`, `TerminatorReturns`, `OverlappedEllipses`, `getCovariance`, and `LinearRegression`.
* The purpose of `counter_cin` and `count_pearson` is not clear from the provided code.
* The function involves complex object tracking and analysis based on shape, size, and orientation.
* The code contains potential optimizations and readability improvements.
"""
function ObjSplitter3D(file, medZ, minZ=0.5 * medZ, maxZ=1.5 * medZ, nCh=1)
    #
    data2d, data, sizesPlanes = data2d_chunking(file, 3, 4, 2, 1, 5, 8, 9, 19, nCh)
    data2d_copy = deepcopy(data2d)
    #
    indexX = 1
    indexY = 2
    indexArea = 3
    indexAng = 4
    indexZ = 5
    indexA = 6
    indexB = 7
    indexType = 8
    indexTag = 8
    indexData = 9
    indexInZ = 10
    indexInRawData = 11
    #
    preok = Vector{}()
    cutPointsF = Vector{}()
    coh = Vector{}()
    val = Vector{}()
    #
    borderEffectLow = 1 #minZ - 1;
    inis = 0
    noInits = zeros(1, 3)
    counter = 0
    counter_cin = 0
    count_pearson = 0
    println("Generating 3D objects")
    for i in tqdm(1:(size(data2d_copy, 1))-borderEffectLow)
        sliceZ = i
        for j in 1:size(data2d_copy[i], 1)
            indexR = j
            goAns = 0 #initiators
            # mean_diameter = sqrt(mean([mean(data2d_copy[i][:,indexArea]) for i in 1:length(data2d_copy)])/pi)*2
            disappear_init = checkIntit(data2d_copy, i, j)
            if any(disappear_init .!= 0)
                noInits = vcat(noInits, disappear_init)
            end
            # println(noInits)
            if data2d_copy[i][j, indexTag] == 0
                inis += 1
                #
                preObj = preObjElongation(sliceZ, indexR, data2d_copy, maxZ, minZ, medZ)
                tempEl = preObj.tempEl
                preVol = preObj.vVolV
                # println("a'' ", tempEl[:,12])
                # println("b'' ", tempEl[:,13])
                distV = preObj.vDist
                angV = preObj.vAng
                pearson = preObj.vPearson
                #
                cutPoint = size(tempEl, 1)
                factor = (pearson)
                if size(tempEl, 1) >= Int(round(minZ)) #&& factor > 0
                    #
                    if size(tempEl, 1) > Int(round(medZ))
                        # Esta es la version original

                        infoTerm = TerminatorReturns(distV[2:end], angV, medZ)
                        cin = infoTerm.numbIn
                        cout = infoTerm.numbOut
                        cin = cin / medZ
                        cout = cout / (size(tempEl, 1) - medZ + 1)
                        cutPV = infoTerm.cutOut
                        lengthPV = infoTerm.lengthOut


                        if cin < cout && cout > 0 #cout > 0
                            counter_cin += 1
                            n_pointH = Int(cutPV[1] + 1) #H has one less element than var
                            #
                            if n_pointH - 2 < 0
                                pearsonIn = 0 #corta por lo que definas antes del cutPoint
                            else
                                pearsonIn = getCovariance(distV[2:n_pointH-1], angV[1:n_pointH-1-1])
                            end
                            #
                            if n_pointH - 1 < 0 #Esto nunca deberia pasar.
                                pearsonOut = 0 #corta por lo que definas antes del cutPoint
                            else
                                pearsonOut = getCovariance(distV[n_pointH:end], angV[n_pointH-1:end])
                            end
                            #
                            if pearsonIn > pearson
                                count_pearson += 1
                                cutPoint = Int(n_pointH)
                                counter += 1
                            end
                            #
                        end

                        # Aquí empiezan las funciones de prueba
                        # cutPoint, doubtedPoint = Termination_Chow(distV,angV)
                        # if cutPoint == 0
                        #     pearsonTot = getCovariance(distV[2:end], angV)
                        #     pearsonTilCut = getCovariance(distV[2:doubtedPoint+1], angV[1:doubtedPoint])
                        #     if pearsonTilCut > pearsonTot
                        #         cutPoint = doubtedPoint
                        #     end
                        # end
                    end
                    #
                    push!(cutPointsF, cutPoint)
                    push!(val, size(tempEl, 1))
                    push!(coh, factor)
                    for zz in 1:cutPoint
                        data2d_copy[Int(tempEl[zz, indexInZ])][Int(tempEl[zz, indexData]), indexTag] += 1
                        tempEl[zz, indexTag] = data2d_copy[Int(tempEl[zz, indexInZ])][Int(tempEl[zz, indexData]), indexTag]
                    end
                    cutTempEl = tempEl[1:cutPoint, :]
                    push!(preok, cutTempEl)
                    #
                end
            end
        end
    end
    k = DataFrame(V=val, C=cutPointsF, pearson=coh)
    #
    return Objects3D(preok, k, inis, noInits), counter_cin, count_pearson
end

"""
ObjSplitter3D_fromBinaryImage(file, nCh=1, zparams::Union{Vector{Int}, Nothing}=nothing)

Splits a 3D binary image into objects based on connected components and shape properties.

**Arguments:**
* `file`: Path to the input binary image file.
* `nCh`: Number of additional channels for object data (default: 1).
* `zparams`: Optional vector of [minZ, medZ, maxZ] values for Z-axis segmentation.

**Returns:**
* `Objects3D`: A struct containing information about the detected objects.
* `counter_cin`: A counter for a specific condition (not fully explained).
* `count_pearson`: A counter for a specific condition (not fully explained).

**Functionality:**
1. Loads the binary image from the file.
2. Uses `regionprops` to extract connected components.
3. Determines Z-axis segmentation parameters if not provided.
4. Converts region properties to a 2D array format using `data2d_fromJuliaRegionProps`.
5. Performs object splitting and analysis using the same logic as `ObjSplitter3D`.
6. Returns an `Objects3D` struct, `counter_cin`, and `count_pearson`.

**Note:**
* The function relies on image processing libraries to extract connected components.
* The `zparams` argument allows for custom Z-axis segmentation.
* The core object splitting logic is shared with the `ObjSplitter3D` function.
* The `counter_cin` and `count_pearson` counters have unclear purposes.
"""
function ObjSplitter3D_fromBinaryImage(img, zparams::Union{Vector{Int},Nothing}=nothing, nCh=1; kernel::Float64 = 6.0)
    #
    # data2d, data, sizesPlanes= data2d_chunking(file, 3, 4, 2, 1, 5, 8, 9, 19, nCh);
    # img = Int.(load(file))
    println("Getting 2D Objects")
    r = regionPropsStack(img, "z", true, kernel = kernel) 
    if typeof(zparams) == Nothing
        println("Generating ellipses")
        e = generateEllipsesImage(r, size(img)[1:2])
        e_im = Gray.(e)
        # Images.save("ellipses.tif", e_im)
        println("Calculating z params")
        xz = regionPropsStack(e[:,round(Int,(size(e,1)/2)-(size(e,3)/2)):round(Int,(size(e,1)/2)+(size(e,3)/2)),:], "y", true)
        # xz = regionPropsStack(e, "y", true)
        zp = min_max_Zdef_fromJuliaRegionProps(xz)
        minZ = floor(zp.minZ/2)
        maxZ = floor(zp.minZ+zp.minZ/2)
        medZ = zp.minZ
        # minZ = floor(Int,zp.medZ); medZ = floor(Int,(zp.medZ+zp.maxZ)/2); maxZ = floor(Int, zp.maxZ)
        println("Autocalculated zparams: ", [minZ, medZ, maxZ])
    else
        minZ = zparams[1]
        medZ = zparams[2]
        maxZ = zparams[3]
    end


    data2d = data2d_fromJuliaRegionProps(r, nCh)
    #println(data2d)
    data2d_copy = deepcopy(data2d)
    #
    indexX = 1
    indexY = 2
    indexArea = 3
    indexAng = 4
    indexZ = 5
    indexA = 6
    indexB = 7
    indexType = 8
    indexTag = 8
    indexData = 9
    indexInZ = 10
    indexInRawData = 11
    #
    preok = Vector{}()
    cutPointsF = Vector{}()
    coh = Vector{}()
    val = Vector{}()
    #
    borderEffectLow = 1 #minZ - 1;
    inis = 0
    noInits = zeros(1, 3)
    counter = 0
    counter_cin = 0
    count_pearson = 0
    println("Generating 3D objects")
    for i in tqdm(1:(size(data2d_copy, 1))-borderEffectLow)
        sliceZ = i
        for j in 1:size(data2d_copy[i], 1)
            indexR = j
            goAns = 0 #initiators
            # mean_diameter = sqrt(mean([mean(data2d_copy[i][:,indexArea]) for i in 1:length(data2d_copy)])/pi)*2
            disappear_init = checkIntit(data2d_copy, i, j)
            if any(disappear_init .!= 0)
                noInits = vcat(noInits, disappear_init)
            end
            # println(noInits)
            if data2d_copy[i][j, indexTag] == 0
                inis += 1
                #
                preObj = preObjElongation(sliceZ, indexR, data2d_copy, maxZ, minZ, medZ)
                tempEl = preObj.tempEl
                preVol = preObj.vVolV
                # println("a'' ", tempEl[:,12])
                # println("b'' ", tempEl[:,13])
                distV = preObj.vDist
                angV = preObj.vAng
                pearson = preObj.vPearson
                #
                cutPoint = size(tempEl, 1)
                factor = (pearson)
                if size(tempEl, 1) >= Int(round(minZ)) #&& factor > 0
                    if size(tempEl, 1) > Int(round(medZ))
                        # Esta es la version original
                        infoTerm = TerminatorReturns(distV[2:end], angV, medZ)
                        cin = infoTerm.numbIn
                        cout = infoTerm.numbOut
                        cin = cin / medZ
                        cout = cout / (size(tempEl, 1) - medZ + 1)
                        cutPV = infoTerm.cutOut
                        lengthPV = infoTerm.lengthOut


                        if cin < cout && cout > 0 #cout > 0
                            counter_cin += 1
                            n_pointH = Int(cutPV[1] + 1) #H has one less element than var
                            #
                            if n_pointH - 2 < 0
                                pearsonIn = 0 #corta por lo que definas antes del cutPoint
                            else
                                pearsonIn = getCovariance(distV[2:n_pointH-1], angV[1:n_pointH-1-1])
                            end
                            #
                            if n_pointH - 1 < 0 #Esto nunca deberia pasar.
                                pearsonOut = 0 #corta por lo que definas antes del cutPoint
                            else
                                pearsonOut = getCovariance(distV[n_pointH:end], angV[n_pointH-1:end])
                            end
                            #
                            if pearsonIn > pearson
                                count_pearson += 1
                                cutPoint = Int(n_pointH)
                                counter += 1
                            end
                            #
                        end

                        # Aquí empiezan las funciones de prueba
                        # cutPoint, doubtedPoint = Termination_Chow(distV,angV)
                        # if cutPoint == 0
                        #     pearsonTot = getCovariance(distV[2:end], angV)
                        #     pearsonTilCut = getCovariance(distV[2:doubtedPoint+1], angV[1:doubtedPoint])
                        #     if pearsonTilCut > pearsonTot
                        #         cutPoint = doubtedPoint
                        #     end
                        # end
                    end
                    #
                    push!(cutPointsF, cutPoint)
                    push!(val, size(tempEl, 1))
                    push!(coh, factor)
                    for zz in 1:cutPoint
                        data2d_copy[Int(tempEl[zz, indexInZ])][Int(tempEl[zz, indexData]), indexTag] += 1
                        tempEl[zz, indexTag] = data2d_copy[Int(tempEl[zz, indexInZ])][Int(tempEl[zz, indexData]), indexTag]
                    end
                    cutTempEl = tempEl[1:cutPoint, :]
                    push!(preok, cutTempEl)
                    #
                end
            end
        end
    end
    for i = 1:length(preok) preok[i][:,11] .= i end
    col_names = vcat([:X,:Y,:Area,:Angle,:Z, :A, :B,:Type,:Tag,:IndexData, :Mask],Symbol.(:Channel,axes(ones(nCh),1)))
    df = DataFrame(vcat(preok...),col_names)
    k = DataFrame(V=val, C=cutPointsF, pearson=coh)
    #
    return Objects3D(preok, k, inis, noInits), df
end

function check_extensions(files::Vector{String})
    extensions = [splitext(file)[2] for file in files]  # Extract extensions
    unique_exts = unique(extensions)
    if length(unique_exts) == 1
        isnotunique = false
        extension = unique_exts[1]
        return extension, isnotunique, files
    else
        ext_counts = Dict(ext => count(==(ext), extensions) for ext in unique_exts)
        majority_ext = argmax(ext_counts)
        isnotunique = true  # Find the most common extension
        files = filter(f -> splitext(f)[2] == majority_ext, files)
        return majority_ext, isnotunique, files  # Return majority extension and its count
    end
end
#Functions to initialize OSCAR


function analyzeImage(path::String, rootpath::String, result_path::String, zparams::Union{Vector{Int},Nothing}=nothing, process::Int=0; kernel::Union{Float64,Nothing}=6.0)
    image::Union{Matrix{Float64},Array{Float64,3},Array{Int,3},Array{Gray{N0f8},3},Array{Gray{N0f16},3}} = Float64.(load(path))
    dims = size(image)
    if process == 1
        println("Processing image...")
        image, kernel = nuclei_processing(image, kernel = kernel)
        image = binarizeImage(image)
        # Images.save("processed_binary.tif", Gray.(image))
    else
        image = binarizeImage(Float64.(image))
    end
    if (typeof(image) == Array{Int,3})
        # println("Kernel size before ObjSplitter = ",kernel)
        objc3d, df = ObjSplitter3D_fromBinaryImage(image,zparams,kernel=kernel)
        summ = summary_objs(objc3d)
        split(basename(path),".")[1]
        # CSV.write("$result_path\\"*split(basename(path),".")[1]*".txt", df, delim = '\t')
        CSV.write("$result_path\\Summary_"*split(basename(path),".")[1]*".txt", summ, delim = '\t')
        println("   >> " * string(length(objc3d.preok)) * " objects found.")
        oscar_output = draw_3D_ellipses_from_summary(summ,dims)
        # overlay = display_overlay_3D(image,Float64.(oscar_output))
        # display(overlay)
        oscar_output = Gray.(oscar_output./ maximum(oscar_output))
        Images.save("$result_path\\"*split(basename(path),".")[1]*".tif",oscar_output)
    else
        r = 0
        println("   (Performing 2D analysis)")
        objects = findObjectsWatershed(image)
        regs = regionProps(objects)
        r += length(regs)
        println("   >> " * string(r) * " objects found.")
    end
end

"""
    create_results_dir(parent::AbstractString, base::String = "exp") -> String

Creates a results subdirectory inside `parent`, named with an incrementing run number and current date.
For example: `exp1_2025-06-05`, `exp2_2025-06-05`, etc.

# Arguments
- `parent::AbstractString`: The directory where the new results folder will be created.
- `base::String`: Prefix for the folder name (default: "exp").

# Returns
- The full path of the created directory as a `String`.
"""
function create_results_dir(parent::AbstractString, base::String = "exp")::String
    today = Dates.format(now(), "yyyy-mm-dd")
    i = 1
    while true
        dir_name = "$(base)$(i)_$today"
        full_path = joinpath(parent, dir_name)
        if !isdir(full_path)
            mkpath(full_path)
            return full_path
        end
        i += 1
    end
    return full_path
end

"""
    draw_3D_ellipses_from_summary(summary_path::String, dims::Tuple{Int, Int, Int}; label_ellipses::Bool=true)

Draws 3D ellipses as 2D cross-sections using the information in a summary file (CSV format).
Each object is decomposed slice by slice into 2D ellipses and rendered into a 3D image stack.
Optionally, each ellipse is labeled with its object ID.

# Arguments
- `summary_path::String`: Path to the summary `.txt` or `.csv` file.
- `dims::Tuple{Int, Int, Int}`: Image dimensions (X, Y, Z).
- `label_ellipses::Bool`: If true, labels each object in the image.

# Returns
- `Array{Int, 3}`: A labeled image stack with the drawn ellipses.
"""
function draw_3D_ellipses_from_summary(summary::DataFrame, dims::Tuple{Int, Int, Int}; label_ellipses::Bool=true)
    img = zeros(Int, dims)
    for row in eachrow(summary)
        cx, cy, cz = row.Xcenter, row.Ycenter, row.Zcenter
        a, b = row.A, row.B
        n_slices = row.Vz
        ang = (-row.AngleXY * pi / 180) +pi/2
        vx, vy = row.Vx, row.Vy
        v_mod = row.C
        label = label_ellipses ? Int(row.Label) : 1

        # Normalize direction
        # norm = sqrt(vx^2 + vy^2 + vz^2)
        # if norm == 0
        #     continue
        # end
        v = [vx, vy] ./ v_mod
        if any(isnan.(v))
            v[findall(isnan.(v))] .= 0
        end

        z_half = floor(Int, n_slices / 2)
        z_start = max(1, round(Int, cz) - z_half)
        z_end = min(dims[3], round(Int, cz) + z_half)
        n_slices = z_end - z_start + 1
        for zi in z_start:z_end
            Δz = zi - cz
            factor = sqrt(1 - (2 * Δz / n_slices)^2)
            a_scaled = a * factor / 2
            b_scaled = b * factor / 2
        
            x = cx + v[1] * Δz
            y = cy + v[2] * Δz
    
            ellipse_coords = ellipse(round(Int, y), round(Int, x), b_scaled, a_scaled, ang)
            valid_coords = filter(p -> all(>(0), Tuple(p)) && p[1] ≤ dims[1] && p[2] ≤ dims[2], ellipse_coords)
            for p in valid_coords
                img[p[1], p[2], zi] = label
            end

        # for zi in 1:dims[3]
        #     Δz = zi - cz
        #     x = cx + v[1] * Δz
        #     y = cy + v[2] * Δz

        #     if abs(Δz) <= (n_slices / 2)  # within bounds of the ellipsoid
        #         factor = sqrt(1 - (2 * Δz / n_slices)^2)
        #         a_scaled = a * factor / 2
        #         b_scaled = b * factor / 2

        #         # Draw the ellipse in slice zi
        #         ellipse_coords = ellipse(round(Int, y), round(Int, x), b_scaled, a_scaled, ang)
        #         valid_coords = filter(p -> all(>(0), Tuple(p)) && p[1] ≤ dims[1] && p[2] ≤ dims[2], ellipse_coords)
        #         for p in valid_coords
        #             img[p[1], p[2], zi] = label
        #         end
        #     end
        end
    end
    # img = Gray.(img./ maximum(img))
    return img
end

function file_folder_selection_GPT()
    win = GtkWindow("Welcome to OSCAR", 500, 100)
    vbox = GtkBox(:v, 10)

    label = GtkLabel("Please provide:\n • A TIFF image in Grayscale with Gray LUT\n\n • OR a TXT file from Fiji script 'get_txt_files.ijm'")
    set_gtk_property!(label, :wrap, true)
    set_gtk_property!(label, :justify, 0)

    btn_file = GtkButton("Select a single file")
    btn_folder = GtkButton("Select a folder")

    hbox = GtkBox(:h, 10)
    set_gtk_property!(hbox, :homogeneous, true)
    push!(hbox, btn_file)
    push!(hbox, btn_folder)
    push!(vbox, label)
    push!(vbox, hbox)
    push!(win, vbox)
    showall(win)

    choice_channel = Channel{Tuple{Int, String}}(1)  # (is_file, path)
    # submitted = Ref(false)  # ✅ new flag

    signal_connect(btn_file, "clicked") do _
        path = open_dialog("Select your file")
        if path != ""
            println("Selected image: " * path)
            put!(choice_channel, (1, path))
        else
            put!(choice_channel, (3, ""))
        end
        # submitted[] = true
    end

    signal_connect(btn_folder, "clicked") do _
        path = open_dialog("Select your folder", action = GtkFileChooserAction.SELECT_FOLDER)
        if path != ""
            println("Selected folder: " * path)
            put!(choice_channel, (0, path))
        else
            put!(choice_channel, (3, ""))
        end
        # submitted[] = true
    end

    # ✅ NEW: Handle window close (e.g., user clicks 'X')
    signal_connect(win, "destroy") do _
        # if !submitted[] && !isready(choice_channel)
        if !isready(choice_channel)
            put!(choice_channel, (3, ""))  # 3 means cancelled
        end
    end

    # ⏳ Block here AFTER everything is shown
    choice = take!(choice_channel)
    destroy(win)
    return choice
end


function get_file_type()
    is_file, path = file_folder_selection_GPT()
    # path::String = ""
    file_type::Symbol = :unknown
    valid_extensions = [".txt",".tif"]
    files::Union{String,Vector{String}} = ""
    if is_file == 1
        # Determine file type before GUI generation
        if endswith(lowercase(path), ".tif")
            file_type = :tiff
        elseif endswith(lowercase(path), ".txt")
            file_type = :txt
        else
            warn_dialog("Unsupported file type. Please select a TIFF or TXT file.")
            return
        end
        files = path
    elseif is_file == 0
        extension, isnotunique, files = check_extensions(readdir(path))
        if !(extension in valid_extensions)
            warn_dialog("File extension, $extension, not supported.")
            return
        else
            if isnotunique 
                warn_dialog("Not all files have the same extension, only analyzing files with $extension extension")
            end
            if extension == ".txt"
                file_type = :txt
            elseif extension == ".tif"
                file_type = :tiff 
            end       
        end  
    elseif is_file == 3
        println("Nor file or folder selected. Exit")
    end
    return file_type, is_file, path, files
end


function startOSCAR()
    file_type, is_file, path, files = get_file_type()
    if is_file == 0 || is_file == 1    # Step 2: GUI - Create window with checkboxes and input fields
        win = GtkWindow("OSCAR Options", 400, 250)
        vbox = GtkBox(:v)

        # Processing checkbox (only for TIFF files)
        process_checkbox = GtkCheckButton("Process the image")
        if file_type == :tiff
            set_gtk_property!(process_checkbox, :active, true)
            instruction_label_process = GtkLabel("Do you want to process the image? Check if yes.")
            # Kernel size options (only for TIFF and if processing is enabled)
            kernel_auto_checkbox = GtkCheckButton("Autocalculate kernel size")
            set_gtk_property!(kernel_auto_checkbox, :active, true)
            label_manual_kernel = GtkLabel("Manual kernel size:")
            entry_kernel = GtkEntry()
            set_gtk_property!(entry_kernel, :text, "6")  # Default value = 3
            set_gtk_property!(label_manual_kernel, :visible, false)
            set_gtk_property!(entry_kernel, :visible, false)
            function on_kernel_auto_checkbox_toggled(widget)
                is_checked = get_gtk_property(widget, :active, Bool)
                set_gtk_property!(label_manual_kernel, :visible, !is_checked)
                set_gtk_property!(entry_kernel, :visible, !is_checked)
            end

        
            signal_connect(on_kernel_auto_checkbox_toggled, kernel_auto_checkbox, "toggled")

            function on_process_checkbox_toggled(widget)
                is_checked = get_gtk_property(widget, :active, Bool)

                # Show/hide autocalculate checkbox
                set_gtk_property!(kernel_auto_checkbox, :visible, is_checked)

                if is_checked
                    # If processing, show manual entry only if autocalculate is unchecked
                    on_kernel_auto_checkbox_toggled(kernel_auto_checkbox)
                else
                    # If not processing, autocalculate is irrelevant; show manual entry
                    set_gtk_property!(label_manual_kernel, :visible, true)
                    set_gtk_property!(entry_kernel, :visible, true)
                end
            end
            signal_connect(on_process_checkbox_toggled, process_checkbox, "toggled")
            push!(vbox, instruction_label_process)
            push!(vbox, process_checkbox)
            push!(vbox, kernel_auto_checkbox)
            push!(vbox, label_manual_kernel)
            push!(vbox, entry_kernel)
        end

        # Possible multichannel stack (only for TXT files)
        nchannels_checkbox = GtkCheckButton("Multichannel stack")
        if file_type == :txt
            set_gtk_property!(nchannels_checkbox, :active, false)
            instruction_label_nchannels = GtkLabel("Does your image have multiple channels? Check if yes.")
            push!(vbox, instruction_label_nchannels)
            push!(vbox, nchannels_checkbox)
        
            label_nch = GtkLabel("Number of channels:")
            entry_nch = GtkEntry()
        
            # Initially hidden
            set_gtk_property!(label_nch, :visible, false)
            set_gtk_property!(entry_nch, :visible, false)
        
            push!(vbox, label_nch)
            push!(vbox, entry_nch)
        
            function on_nchannels_checkbox_toggled(widget)
                is_checked = get_gtk_property(widget, :active, Bool)
                set_gtk_property!(label_nch, :visible, is_checked)
                set_gtk_property!(entry_nch, :visible, is_checked)
            end
            on_nchannels_checkbox_toggled(nchannels_checkbox)
            signal_connect(on_nchannels_checkbox_toggled, nchannels_checkbox, "toggled")
        end

        # Z-parameter options (always available)
        zparam_checkbox = GtkCheckButton("Use autocalculated zparams")
        set_gtk_property!(zparam_checkbox, :active, true)
        # instruction_label = GtkLabel("Tick the box to use autocalculated zparams, or leave unticked to provide manually.")
        instruction_label = GtkLabel("Do you want to autocalculate the zparams?. Check if yes")
        push!(vbox, instruction_label)
        push!(vbox, zparam_checkbox)

        # Manual zparam input fields (initially hidden)
        label_manual_zparams = GtkLabel("Provide manual zparams:")
        entry_z1, entry_z2, entry_z3 = GtkEntry(), GtkEntry(), GtkEntry()
        label_min_z, label_med_z, label_max_z = GtkLabel("Min Z:"), GtkLabel("Med Z:"), GtkLabel("Max Z:")

        for widget in [label_manual_zparams, label_min_z, entry_z1, label_med_z, entry_z2, label_max_z, entry_z3]
            set_gtk_property!(widget, :visible, false)
        end

        function on_zparam_checkbox_toggled(widget)
            is_checked = get_gtk_property(widget, :active, Bool)
            for widget in [label_manual_zparams, label_min_z, entry_z1, label_med_z, entry_z2, label_max_z, entry_z3]
                set_gtk_property!(widget, :visible, !is_checked)
            end
        end
        
        signal_connect(on_zparam_checkbox_toggled, zparam_checkbox, "toggled")

        button_proceed = GtkButton("Proceed")

        function on_button_proceed_clicked(widget)
            process = (file_type == :tiff && get_gtk_property(process_checkbox, :active, Bool)) ? 1 : 0
            # destroy(win)
            nch = nothing
            if get_gtk_property(nchannels_checkbox, :active, Bool)
                nch = parse(Int, get_gtk_property(entry_nch, :text, String))
            else
                nch = 1
            end


            zparams = nothing

            if !get_gtk_property(zparam_checkbox, :active, Bool)
                # Explicitly get text from GtkEntry fields
                z1_text = get_gtk_property(entry_z1, :text, String) |> strip
                z2_text = get_gtk_property(entry_z2, :text, String) |> strip
                z3_text = get_gtk_property(entry_z3, :text, String) |> strip
        
                # println("Retrieved input values: z1='$z1_text', z2='$z2_text', z3='$z3_text'")  # Debugging
        
                # Check if any field is empty
                if isempty(z1_text) || isempty(z2_text) || isempty(z3_text)
                    warn_dialog("Please provide values for all Z parameters.")
                    return  # Stop execution if fields are empty
                end
        
                try
                    z1 = parse(Int, z1_text)
                    z2 = parse(Int, z2_text)
                    z3 = parse(Int, z3_text)
                    zparams = [z1, z2, z3]
                    println("Manual zparams provided: $zparams")
                catch
                    warn_dialog("Invalid Z parameter input. Please enter valid numbers.")
                    return
                end
            else
                println("Using autocalculated zparams")
                zparams = nothing  # Placeholder for autocalculated values
            end

            kernel_size = nothing
            if file_type == :tiff
                if get_gtk_property(process_checkbox, :active, Bool)
                    if !get_gtk_property(kernel_auto_checkbox, :active, Bool)
                        kernel_text = strip(get_gtk_property(entry_kernel, :text, String))
                        if isempty(kernel_text)
                            warn_dialog("Please provide a kernel size or choose autocalculation.")
                            return
                        end
                        try
                            kernel_size = parse(Float64, kernel_text)
                        catch
                            warn_dialog("Invalid kernel size input. Please enter a valid number.")
                            return
                        end
                    else
                        println("Kernel size will be autocalculated.")
                    end
                else
                # Not processing: Ask for approximate kernel size
                    kernel_text = strip(get_gtk_property(entry_kernel, :text, String))
                    if isempty(kernel_text)
                        warn_dialog("Please provide the approximate kernel size you used.")
                        return
                    end
                    try
                        kernel_size = parse(Float64, kernel_text)
                    catch
                        warn_dialog("Invalid approximate kernel size. Please enter a valid number.")
                        return
                    end
                end
                println("Kernel size set to: ", kernel_size)
            end

            destroy(win)

            if is_file == 1  # Single file selected
                if file_type == :txt
                    base_dir = dirname(dirname(path))
                    results_dir = create_results_dir(base_dir)
                    base_name = splitext(basename(path))[1]
                    if zparams == nothing
                        xz_folder = joinpath(base_dir, "xz_files")
                        matching_txt = joinpath(xz_folder, base_name * ".txt")
                        if isfile(matching_txt)
                            println("Found related TXT file in xz_files: " * matching_txt)
                            xz_path = matching_txt
                            zp = min_max_Zdef(xz_path)
                            minZinterval = floor(zp.minZ/2); maxZinterval = floor(zp.minZ+zp.minZ/2); medZinterval = zp.minZ;
                            println("Autocalculated zparams: ", [minZinterval, medZinterval, maxZinterval])
                        else
                            println("No related TXT file found in xz_files.")
                        end
                    else
                        minZinterval = zparams[1]; medZinterval = zparams[2]; maxZinterval = zparams[3]
                    end
                    
                    if minZinterval < 1
                        println("z param tends to 1. This value means that all 2D objects are 3D objects.")
                    else
                        objc3d, counter_cin, counter_pearson = ObjSplitter3D(path, medZinterval, minZinterval, maxZinterval, nch)
                        summary_df = summary_objs(objc3d, nch)
                        CSV.write("$results_dir\\Summary_$base_name.txt", summary_df, delim = '\t')
                        println("   >> " * string(length(objc3d.preok)) * " objects found.")
                        println("Analysis completed!")
                    end
            
                elseif file_type == :tiff
                    # try
                        println("Analyzing: " * path)
                        results_dir = create_results_dir(dirname(path))
                        println("Kernel size before analyzeImage = ",kernel_size)
                        analyzeImage(path, dirname(path), results_dir, zparams, process, kernel = kernel_size)
                        println("Analysis completed!")
                    # catch
                    #     warn_dialog("Oops! Something went wrong during analysis...")
                    # end
                end
            elseif is_file == 0  # Folder selected
                if file_type == :txt
                    base_dir = dirname(path)
                    results_dir = create_results_dir(base_dir)
                    for file in files
                        file_path = joinpath(path, file)
                    # if endswith(lowercase(file), ".txt")
                        base_name = splitext(file)[1]
                        if zparams == nothing
                            xz_folder = joinpath(base_dir, "xz_files")
                            matching_txt = joinpath(xz_folder, base_name * ".txt")
                            if isfile(matching_txt)
                                println("Found related TXT file in xz_files: " * matching_txt)
                                xz_path = matching_txt
                                zp = min_max_Zdef(xz_path)
                                minZinterval = floor(zp.minZ/2); maxZinterval = floor(zp.minZ+zp.minZ/2); medZinterval = zp.minZ;
                                println("Autocalculated zparams: ", [minZinterval, medZinterval, maxZinterval])
                            else
                                println("No related TXT file found in xz_files.")
                            end
                        else
                            minZinterval = zparams[1]; medZinterval = zparams[2]; maxZinterval = zparams[3]
                        end
                
                        if minZinterval < 1
                            println("z param tends to 1. This value means that all 2D objects are 3D objects.")
                        else
                            objc3d, counter_cin, counter_pearson = ObjSplitter3D(file_path, medZinterval, minZinterval, maxZinterval, nch)
                            summary_df = summary_objs(objc3d, nch)
                            CSV.write("$results_dir\\Summary_$base_name.txt", summary_df, delim = '\t')
                            println("   >> " * string(length(objc3d.preok)) * " objects found.")
                            println("Analysis completed!")
                        end
                    end
                elseif file_type == :tiff
                    results_dir = create_results_dir(path)
                    for file in files
                        file_path = joinpath(path, file)
                        println("Analyzing: " * file)
                        analyzeImage(file_path, path, results_dir, zparams, process, kernel = kernel_size)
                        println("Analysis completed!")
                    end
                end
            end
        end

        signal_connect(on_button_proceed_clicked, button_proceed, "clicked")

        for widget in [label_manual_zparams, label_min_z, entry_z1, label_med_z, entry_z2, label_max_z, entry_z3, button_proceed]
            push!(vbox, widget)
        end
        push!(win, vbox)

        showall(win)
        
        on_zparam_checkbox_toggled(zparam_checkbox)
        if file_type == :tiff
            on_kernel_auto_checkbox_toggled(kernel_auto_checkbox)
        end
        if file_type == :txt
            on_nchannels_checkbox_toggled(nchannels_checkbox)
        end
    end
    return nothing
end
