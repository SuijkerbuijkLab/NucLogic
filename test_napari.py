import tifffile
import random
import os
input = r"E:\Users\Sebastian_van_Dijk\Data_Anna\s65\cropped"
nuclei = 2

for file in os.listdir(input):
    image = os.path.join(input, file)
    image = tifffile.imread(image)[nuclei,:,:,:]

    z, y, x = image.shape
    numbers = random.sample(range(0, z), 4)
    
    for number in numbers:
        print(number)
        plane = image[number,:,:]
        tifffile.imwrite(os.path.join(os.path.dirname(input), "training", f"{file.split('.')[0]}_{number}.tif"), plane)
