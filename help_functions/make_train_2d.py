import tifffile
import numpy as np
import os

output = r"C:\Users\6331823\Local SSD\Train dual nuclei 20x"
image = tifffile.imread(
    r"C:\Users\6331823\Local SSD\Data_Elise\EM_Exp007_Imaging_20x_2025-11-21_Maria_Elise_14.47.50_CZC34578YN_F24\cropped\Frame-0.tif"
)

nuclei = image[:, 0:1]

nuclei = np.max(image, axis=1)

# select 10 random z slices and save them to output folder

# Select 10 random z slices
n_slices = 10
max_z = nuclei.shape[0]
random_indices = np.random.choice(max_z, size=min(n_slices, max_z), replace=False)
random_indices = np.sort(random_indices)  # Sort for easier review

print(f"Selected z-slices: {random_indices}")

# Save each slice
for z_idx in random_indices:
    slice_img = nuclei[z_idx]
    output_path = os.path.join(output, f"Frame-0_z{z_idx:03d}.tif")
    tifffile.imwrite(output_path, slice_img)
    print(f"Saved: {output_path}")

print(f"\nSaved {len(random_indices)} slices to {output}")
