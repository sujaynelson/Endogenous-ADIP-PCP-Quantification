## Quantitative Image Analysis Pipeline

This repository contains the quantitative image analysis pipeline used in the study:

**_Planar polarization of endogenous ADIP during Xenopus neurulation_**

Image quantification was performed using the GitHub repository:  
https://github.com/JakobSchauser/Mechanical-cues-organize-planar-cell-polarity.git  

with the modifications described below.


---

## Image Quantification Steps

### 1. Cell Segmentation
Segment cells in confocal images using [Cellpose](https://www.cellpose.org) in a Terminal/Anaconda environment.

### 2. Extract Cell Masks
Extract masks for each cell using [read_numpy_data.py](read_numpy_data.py).

### 3. Organize Data
Place the following files together in the `all_data_structured` folder:
- Image file
- Corresponding mask file (named `mask_filename`)
- `channels.txt` file describing the channels

### 4. Detect Features
Detect ADIP aggregates, bicellular junctions, and tricellular junctions using [cell_strain_from_image.py](cell_strain_from_image.py).

### 5. Generate Rose Plots
- Rose plots representing planar polarity were generated using [Creating_figures.py](Creating_figures.py)

### 6. Generate Combined Rose Plots
- For combined rose plots from multiple embryo images, use [generate_combined_roseplot.py](generate_combined_roseplot.py)

---

## Channel Configuration

In `channels.txt`, define the ADIP, cell wall, and Actin channels in that order, separated by newlines (0-indexed). Actin is a stand-in for any non-aggregating proteins.

### Examples

**ADIP in channel 2, cell wall in channel 1:**
```
1
0
```

**ADIP in channel 1, cell wall in channel 2, Actin in channel 3:**
```
0
1
2
```

**Only cell wall in channel 1 (no ADIP or Actin):**
```
-1
0
-1
```

---

## Additional Tools

- [aggregate_analysis_tools.py](aggregate_analysis_tools.py) - Tools for creating rose plots and other analyses
- [strain_and_adip_tools/](strain_and_adip_tools/) - Core analysis modules

