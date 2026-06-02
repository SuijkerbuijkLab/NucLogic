# NucLogic

We present NucLogic, a fast and robust 3D nuclear segmentation algorithm that combines the maturity of 2D segmentation with geometric stitching rules based on nuclear shape, orientation, and intensity to reconstruct accurate 3D volumes. NucLogic reliably segments challenging datasets including long-term timelapse imaging of organoids, and large dense structures such as liver microtissues. Benchmarked against Cellpose 4, NucLogic matches F1 scores at high SNR, with outperformance increasing as SNR decreases. Segmentation completes at 20–70% of the computation time, with greater speed gains on larger images. Beyond segmentation, NucLogic includes an intuitive graphical user interface enabling non-experts to segment, quantify, and visualize 3D and 4D data. This allows users to quantify how many cells are present, what cell- or phenotype they are, what their shape and density is, and how they interact with their neighbours.

#### The overall NucLogic pipeline:
![alt text](miscellaneous/pipeline_overview.png)

#### Example organoid timelapse showing segmentation and called phenotypes
https://github.com/user-attachments/assets/044b71ff-695d-489e-83bf-30c1071486dc


<br>

# Installation instructions

1. Make sure you have Git installed on your computer (download [here](https://git-scm.com/install/)).  

2. Get the repository using one of the two options below:

   **Option A — Clone with git (recommended, models download automatically):**  
   Make sure [git](https://git-scm.com/downloads) is installed on your computer. Open a terminal in the folder where you want to install the software (on windows, use file explorer to navigate to a install location of your choice, right click, and select "open in terminal") and run:
   ```
   git clone https://github.com/SebastianVanDijk/organoid_segmenter.git
   ```
   This downloads everything including the model files into a new *NucLogic* folder.

   **Option B — Download as zip (download models manually):**  
   Use the green code button above to download the zip and extract it to a folder of your choice. Then manually download the raw model files by going into the *models* folder on this page, opening each of *2d_high_quality_model*, *2d_low_quality_model*, *sam2.1_hiera_s.yaml* and *sam2.1_hiera_small.pt* and clicking *Download raw file*. Place these in the *models* folder you extracted in this step.

3. **Windows:** Start the software by double clicking **NucLogic.bat**. You can create a shortcut of the NucLogic.bat file and save this on your desktop for easy access.  
   **Linux:** Run **Linux_NucLogic.sh** from a terminal (`bash Linux_NucLogic.sh`).  
   The first time you do this it will take some time, as it sets up the environment automatically. You will use the same file to start the app next time.

**Note:**  
If the app does not start, check the *logs* folder in the installation directory for error details.   
NucLogic uses Cellpose Sam, which is extremely slow on cpu. Please use an NVIDIA GPU for segmentation. NucLogic uses CUDA 12.8 for GPU acceleration, which requires an NVIDIA GPU with compute capability 3.5 or higher (most GPU's released from 2012 onwards). You can check if your GPU is compatible [here](https://developer.nvidia.com/cuda/gpus). Make sure your computer has the latest GPU drivers installed.


<br>

# Using NucLogic

**Input files:**  
Both tif and ims files can be used as input for 3D or 4D imaging data. NucLogic reads voxel size and timepoint metadata from the file where available, and falls back to default values otherwise. Timelapses should be saved as a single file, not split per timeframe.

**Preparing your data:**  
Place all files you want to analyse into a single folder and navigate to it in NucLogic. You will be prompted to generate a subfolder for each sample — this is required, as NucLogic expects each sample in its own folder to keep output files organised.

**Segmenting:**  
Select the samples you want to analyse and go to the **Segment** tab. Fill in all required information, making sure to specify all channels present in your images. You can then run the full pipeline for segmentation and analysis. At later timepoints you can skip the segmentation step and only rerun the analysis, which saves siNucLogic can use tif and ims files as input to segment 3D or 4D imaging data. NucLogic tries to read image metadata for voxel size and timepoints, but falls back to default values when this is not found in the input file. Timelapses should be saved as a single file, not seperated per time frame. Place all the files you want to analyse into a folder, and navigate to this folder in NucLogic. You will be prompted to generate directories for your samples, this is done because NucLogic expexts every sample in a seperate folder, to prevent cluttering of all of the output files generated. Next, select the samples you want to analyse and move to the "segment" tab. Here, fill out all the necessary information. Please inform NucLogic of all the channels present in your images. You can then choose to start the full segmentation pipeline for segmentation and analysis. If you have previously segmented these samples but want to generate more statistics, you can also choose to run parts of the pipeline and skip the segmentation step, in order to save a lot of computing time.. 
Save and load configurations for replicatability in the settings between experiments.
In the advanced segmentations settings, you can also upload your own 2D cellpose SAM models for 2D segmentation.
To analyse images with different sets of imaged channels, please divide this over seperate segmentation runs, as NucLogic cannot handle this.
The view data tab can be used to open and inspect your samples and segmentations in Napari.
The export data tab can be used to generate tsv files (can be opened in excel or any programming language) of the generated data of all your selected samples.gnificant computing time while still generating up-to-date statistics.

For images with different sets of channels, run NucLogic separately for each group — mixed channel sets within a single run are not supported.

**Settings:**  
Save and load your configuration between experiments for reproducibility. In the advanced segmentation settings you can also supply your own 2D Cellpose SAM models.

**Viewing and exporting:**  
Use the **View data** tab to open and inspect your samples and segmentations in Napari. Use the **Export data** tab to generate TSV files of the analysis results for all selected samples, which can be opened in Excel or any programming language.

<br>

# References
Developed by Sebastian van Dijk in the Suijkerbuijk lab at Utrecht University, Department of Developmental Biology. Based on initial ideas by Mario Ledesma Terrón, who published his OSCAR stitching logic:
- Object Stitching by Clustering of Adjacent Regions for accurate quantification of three-dimensional tissues;
Mario Ledesma-Terrón, Diego Pérez-Dones, Diego Mazo-Durán, Gemma Navarro-Martinez, Gonzalo G. Gíron, David G. Míguez;  J Cell Sci 15 September 2025; 138 (18): jcs264316. doi: https://doi.org/10.1242/jcs.264316

Also makes use of Cellpose 4 and Meta SAM:
- Cellpose-SAM: superhuman generalization for cellular segmentation;
Marius Pachitariu, Michael Rariden, Carsen Stringer;
bioRxiv 2025.04.28.651001; doi: https://doi.org/10.1101/2025.04.28.651001
- SAM 2: Segment Anything in Images and Videos;
 Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and Radle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Dollar, Piotr and Feichtenhofer, Christoph; 
 arXiv preprint arXiv:2408.00714; url=https://arxiv.org/abs/2408.00714; 2024

