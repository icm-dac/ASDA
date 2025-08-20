# Automated Spot Detection and Analysis

<img src="asda_logo.png" width=300 alt="ASDA Logo">

## Status: Under Review for Publication

This repository contains code that is currently under review for publication. Please read the LICENSE file carefully.

## Overview

This pipeline automatically detects, classifies, and analyzes colored spots in microscopy images. Originally developed for biological imaging applications where precise spot counting and spatial analysis are critical, it provides a comprehensive GUI-based solution using HSV color space analysis with advanced clustering capabilities.

## Features

- **Interactive GUI** built with PyQt5 for straightforward analysis
- **Multi-color detection** across 24+ predefined colors using HSV color space
- **Advanced clustering** system that groups spots by color similarity
- **Interactive ROI selection** for marking injection sites
- **Automatic preprocessing** with foreground extraction and bounding box detection
- **Comprehensive statistics** including distances and neighbor relationships
- **Excel export** with detailed measurements
- **Rich visualizations** with annotated images, arrows, and overlays
- **Statistical analysis** with violin plots and significance testing

## Installation

### Dependencies
```bash
pip install PyQt5 opencv-python numpy pandas matplotlib seaborn scikit-image scipy scikit-learn tqdm
```

### Requirements
- **Python**: 3.8+
- **Supported image formats**: .png, .jpg, .jpeg, .tif, .tiff
- **Input images**: RGB format (RGBA automatically converted)
- **Recommendation**: High contrast images for optimal detection

## Usage

### Quick Start
```bash
python main.py
```

### Step-by-Step Process
1. **Load Image**: Browse and select your microscopy image
2. **Set Output**: Choose destination directory for results  
3. **Configure Parameters**: Adjust settings as needed (see Configuration below)
4. **Mark Injection Site**: Draw rectangle around injection point and confirm
5. **Run Analysis**: Click "Run Analysis" and monitor progress
6. **View Results**: Visualizations appear in right panel when complete

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Key Fraction | 0.25 | Proportion of image width used for color key detection |
| Buffer | 50 | Pixel buffer around key region |
| K-Means Clusters | 0 | Number of clusters for color grouping (0 = disabled) |
| Debug Mode | Off | Enable detailed diagnostic output |

## Output

### File Structure
```
output_directory/[image_name]/
├── [image_name]_analysis_results.xlsx
├── [image_name]_overview_arrows.png
├── [image_name]_[color]_spots.png
├── [image_name]_[color]_arrows.png
├── [image_name]_distance_distribution.svg
├── [image_name]_neighbor_distribution.svg
└── [image_name]_same_color_distribution.svg
```

### Data Export (Excel)
Each detected spot includes:
- **Spot_ID**: Unique identifier within color group
- **Centroid_X/Y**: Pixel coordinates of spot center
- **Area**: Spot area in pixels
- **Distance_to_Injection**: Euclidean distance from injection site
- **Nearest_Neighbor_Distance**: Distance to closest spot (any color)
- **Nearest_Neighbor_Color**: Color of nearest neighboring spot
- **Nearest_Same_Color_Distance**: Distance to nearest same-color spot
- **Intensity**: Average pixel intensity of spot region

### Visualizations
- **Spot plots**: Colored circles with IDs (with/without arrows to injection site)
- **Overview image**: All colors displayed together with arrows
- **Statistical plots**: Violin plots with Mann-Whitney U significance testing

## Performance & Limitations

### Performance
- Memory usage scales with image size
- Processing time depends on image complexity and spot count
- Single-threaded processing

### Current Limitations
- One image at a time (no batch processing)
- High memory usage with very large images
- Manual injection site selection required
- Processing time increases with spot density

## License

This code is under a temporary license during the review period. See the LICENSE file for the full terms.
