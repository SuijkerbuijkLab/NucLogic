# NucLogic

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.10.05.756154-b31b1b)](https://doi.org/10.64898/2026.10.05.756154)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23110801.svg)](https://doi.org/10.5281/zenodo.23110801)
[![Models DOI](https://img.shields.io/badge/models%20DOI-10.5281%2Fzenodo.23110965-blue)](https://doi.org/10.5281/zenodo.23110965)

Paper on bioRxiv: [10.64898/2026.10.05.756154](https://doi.org/10.64898/2026.10.05.756154). Software archived at [10.5281/zenodo.23110801](https://doi.org/10.5281/zenodo.23110801); pretrained models archived at [10.5281/zenodo.23110965](https://doi.org/10.5281/zenodo.23110965). If you use NucLogic, please cite the paper (see [Citation](#citation)).

We present NucLogic, a fast and robust 3D nuclear segmentation algorithm that combines the maturity of 2D segmentation with geometric stitching rules based on nuclear shape, orientation, and intensity, to reconstruct accurate 3D volumes. NucLogic reliably segments challenging datasets including long-term timelapse imaging of organoids, and large dense structures such as the zebrafish brain. Benchmarked against other state-of-the-art segmentation tools, NucLogic matches F1 scores at high SNR, with increasing outperformance as SNR decreases. Importantly, segmentation completes at 20–40% of the computation time, with greater speed gains on larger images. Beyond segmentation, NucLogic includes an intuitive graphical user interface allowing non-experts to segment, quantify, and visualize 3D and 4D data. This enables straightforward quantification of cell numbers, alongside integration of a wide range of both cell-intrinsic properties, such as phenotype and morphology, and local environment metrics, such as density and spatial interactions. Together, NucLogic makes quantitative 3D nuclear analysis of previously unmanageable datasets accessible to a broad research community. 

#### The overall NucLogic pipeline:
![alt text](miscellaneous/pipeline_overview.png)

#### Example organoid timelapse showing segmentation and called phenotypes
https://github.com/user-attachments/assets/044b71ff-695d-489e-83bf-30c1071486dc


<br>

# Installation instructions

1. Make sure you have Git installed on your computer (download [here](https://git-scm.com/install/)).  

2. Get the repository using one of the two options below:

   **Option A — Clone with git (recommended):**  
   Open a terminal in the folder where you want to install the software (on Windows, use File Explorer to navigate to an install location of your choice, right click, and select "open in terminal") and run:
   ```
   git clone https://github.com/SuijkerbuijkLab/NucLogic.git
   ```

   **Option B — Download as zip:**  
   Use the green code button above to download the zip and extract it to a folder of your choice.

3. **Windows:** Go into the installed "NucLogic" folder and start the software by double clicking **NucLogic.bat**. On the first launch you'll be asked for administrator rights. These are only needed to make NucLogic available to every user on the computer; if you decline, it installs for your own account only. You can create a shortcut of the NucLogic.bat file and save this on your desktop for easy access.  
   **Linux:** Run **Linux_NucLogic.sh** from a terminal (`bash Linux_NucLogic.sh`).  
   **Mac:** Run **NucLogic.command** by double clicking it. Only Apple silicon Macs are supported, not Intel.
   The first time you do this it will take some time, as it sets up the environment automatically. You will use the same file to start the app next time.  
   On this first launch NucLogic also downloads the model weights (~1.4 GB), showing progress in a bar at the top of the window. This happens only once.
   Additionally, NucLogic automatically checks for updates and will prompt ask you to update the software every time a new version is released.

**Note:**  
If the app does not start, check the *logs* folder in the installation directory for error details.   
NucLogic uses Cellpose-SAM, which is extremely slow on CPU. Please use an NVIDIA GPU for segmentation. NucLogic uses CUDA 12.8 for GPU acceleration, which requires an NVIDIA GPU with compute capability 7.0 or higher (for example GeForce GTX 16 / RTX 20 series and newer). You can check your GPU's compute capability [here](https://developer.nvidia.com/cuda/gpus). Make sure your computer has the latest GPU drivers installed. On Apple silicon Macs, NucLogic uses the built-in Apple GPU automatically (MPS).


<br>

# Using NucLogic

**Input files:**  
NucLogic reads 3D and 4D imaging data in the following formats:

| Format | Extension |
| --- | --- |
| TIFF (OME-TIFF, ImageJ, plain) | `.tif`, `.tiff`, `.ome.tif` |
| Imaris | `.ims` |
| Nikon* | `.nd2` |
| Zeiss* | `.czi`, `.lsm` |
| Leica* | `.lif` |
| OME-Zarr / NGFF | `.zarr`, `.ome.zarr` |
| MetaMorph | `.nd`, `.STK (per channel)` |

Voxel size and timepoint interval are read from the file where available, and fall back to the values set in the advanced settings otherwise. Timelapses should be saved as a single file, not split per timeframe.

\* ND2, CZI and LIF files can hold several stage positions or scenes in one container. NucLogic detects these and offers to write each position out as its own sample, so that a whole plate can be analysed rather than just its first position. The original file is left untouched, and you are asked afterwards whether to delete it.

**Preparing your data:**  
Place all files you want to analyse into a single folder and navigate to it in NucLogic. You will be prompted to generate a subfolder for each sample — this is required, as NucLogic expects each sample in its own folder to keep output files organised.

**Multiposition files:**  
ND2, CZI and LIF files can hold several stage positions or scenes in one container. NucLogic detects these and offers to write each position out as its own sample, so that a whole plate can be analysed rather than just its first position. Each position becomes a folder holding one OME-TIFF, named after the scene where the file provides a name. The original file is left untouched, and you are asked afterwards whether to delete it. The conversion can be cancelled at any point; positions already written stay usable and re-running continues where it stopped.

**Segmenting:**  
Select the samples you want to analyse and go to the **Segment** tab. Fill in all required information, making sure to specify all channels present in your images. You can then run the full pipeline for segmentation and analysis. If you have previously segmented these samples but want to generate more statistics, you can skip the segmentation step and rerun only the analysis, which saves a lot of computing time. Configurations can be saved and reloaded in the settings, so the same analysis can be repeated across experiments.
In the advanced segmentation settings, you can also upload your own 2D Cellpose-SAM models for 2D segmentation, or download extra pretrained models.
To analyse images with different sets of channels, run NucLogic separately for each group — mixed channel sets within a single run are not supported.

**Settings:**  
Save and load your configuration between experiments or segmentation runs for reproducibility.

**Viewing and exporting:**  
The view data tab can be used to open and inspect your samples and segmentations in napari. A custom napari plugin links each label layer to that sample's segmentation statistics tsv file. Any property can be selected, upon which a histogram and a range slider are shown, and the displayed nuclei or cells are restricted in place to those falling inside every active range. This allows populations to be identified visually and interactively. The resulting selection can be saved as a TIFF, or annotated back into the statistics table as an additional column for downstream use. Use this tool to find the optimal custom cutoff value for phenotype calling!
The export data tab can be used to generate tsv files (can be opened in Excel or any programming language) of the generated data of all your selected samples.

<br>

# Citation
If you use NucLogic in your research, please cite:
- NucLogic: Logic-based geometric 3D nuclear segmentation for fast and robust analysis of challenging biological imaging data;
Sebastian G. van Dijk, Mario Ledesma-Terrón, Susanne J. Kraus, Ruth van Brussel, Emmanuel Marquez-Legorreta, Lars J. S. Kemp, Saskia J. E. Suijkerbuijk;
bioRxiv 2026.10.05.756154; doi: https://doi.org/10.64898/2026.10.05.756154

GitHub's "Cite this repository" button gives the same reference in APA and BibTeX format.

<br>

# References
Developed by Sebastian van Dijk in the Suijkerbuijk lab at Utrecht University, Division of Developmental Biology. Based on initial ideas by Mario Ledesma-Terrón, who published his OSCAR stitching logic:
- Object Stitching by Clustering of Adjacent Regions for accurate quantification of three-dimensional tissues;
Mario Ledesma-Terrón, Diego Pérez-Dones, Diego Mazo-Durán, Gemma Navarro-Martinez, Gonzalo G. Gíron, David G. Míguez;  J Cell Sci 15 September 2025; 138 (18): jcs264316. doi: https://doi.org/10.1242/jcs.264316

Also makes use of Cellpose 4 and Meta SAM:
- Cellpose-SAM: superhuman generalization for cellular segmentation;
Marius Pachitariu, Michael Rariden, Carsen Stringer;
bioRxiv 2025.04.28.651001; doi: https://doi.org/10.1101/2025.04.28.651001
- SAM 2: Segment Anything in Images and Videos;
 Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and Radle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Dollar, Piotr and Feichtenhofer, Christoph; 
 arXiv preprint arXiv:2408.00714; url=https://arxiv.org/abs/2408.00714; 2024

<br>

# License
NucLogic is released under the [MIT License](LICENSE).


The pretrained model weights are distributed under their own licences: the SAM 2.1 checkpoint under the Apache License 2.0, and the Cellpose models under the BSD 3-Clause License of the Cellpose-SAM weights they were trained from. See [models/LICENSE.txt](models/LICENSE.txt). Third-party packages installed with NucLogic remain under their own licences.

Software archived at [10.5281/zenodo.23110801](https://doi.org/10.5281/zenodo.23110801); pretrained models archived at [10.5281/zenodo.23110965](https://doi.org/10.5281/zenodo.23110965).

