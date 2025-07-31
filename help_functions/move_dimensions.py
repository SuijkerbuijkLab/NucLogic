# File that can alter the order of dimensions of a tiff file

import tifffile
import numpy as np

import os

input = r"C:\Users\6331823\Local SSD\Data_Anna\s87\s87.tif"

movie = tifffile.imread(input)

movie = np.transpose(movie, (0, 2, 1, 3, 4))

# Extract shape dimensions
sizeT, sizeC, sizeZ, sizeY, sizeX = movie.shape

# Build OME-XML dynamically
ome_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="dynamic_ome.tif">
    <Pixels DimensionOrder="TCZYX"
            Type="uint16"
            SizeT="{sizeT}" SizeC="{sizeC}" SizeZ="{sizeZ}"
            SizeY="{sizeY}" SizeX="{sizeX}"
            SignificantBits="16"
            BigEndian="false"
            PhysicalSizeX="1.0" PhysicalSizeY="1.0"
            PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm"
            Interleaved="false">
      {''.join([f'<Channel ID="Channel:0:{c}" SamplesPerPixel="1"/>' for c in range(sizeC)])}
    </Pixels>
  </Image>
</OME>
"""


def sanitize_ascii(xml_string):
    return xml_string.encode("ascii", "ignore").decode("ascii")


tifffile.imwrite(
    os.path.join(os.path.dirname(input), "new.tif"),
    movie,
    photometric="minisblack",
    description=sanitize_ascii(ome_xml),
    metadata=None,
)
