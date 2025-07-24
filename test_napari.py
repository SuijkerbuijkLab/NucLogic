import tifffile
import numpy as np
input = r"E:\Users\Sebastian_van_Dijk\Data_Anna\s65\s65.tif"

movie = tifffile.imread(input)

# Reorder movie from (T, C, Z, Y, X) to (Z, Y, X, C, T)
movie_reordered = np.transpose(movie, (0,2,1,3,4))

tifffile.imwrite(f"{input}-2", movie_reordered)