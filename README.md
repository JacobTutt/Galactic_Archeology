# Mapping the Galactic Halo: Automated Detection of Substructure in Gaia DR3

## Author: Jacob Tutt, Department of Physics, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Descriptions

This project investigates the structure of the Milky Way’s stellar halo using Gaia EDR3 astrometry, with the goal of identifying substructures such as globular clusters and tidal streams through automated, data-driven techniques.

1.	**Data Preprocessing**
- Optimises photometric, parallax, and proper motion quality cuts using benchmark clusters to enhance the visibility of substructure populations.

2.	**Halo Mapping**
- Produces full-sky RGB and density maps to visualise stellar overdensities and guide subsequent algorithmic detection strategies.
	
3.	**Substructure Detection**
- Applies both traditional (HDBSCAN, Extreme Deconvolution) and novel clustering methods in proper motion and integral-of-motion space to recover halo substructures, including globular clusters and tidal streams.

4.	**Chemical Validation**
- Integrates APOGEE DR17 spectroscopic data to examine the chemical distinctiveness of recovered populations and provide insight into the Milky Way’s chemical evolution.

This repository forms part of the submission for the MPhil in Data Intensive Science's A5 Galactic Archeology Course at the University of Cambridge.

## Table of Contents
- [Data](#data)
- [Notebooks](#notebooks)
- [Results](#results)
- [Installation](#installation-and-usage)
- [License](#license)
- [Support](#support)
- [Author](#author)


## Data
During this project, data was aquired from the [Gaia ESA Archive](https://gea.esac.esa.int/archive/) using ADQL queries. To allow for reproducibility, the [data file](data/0_Data.md) provides an outline of the raw files that were stored within the [data directory](data) and used as well as the motivation behind them and the necessary queries to recreate them.

Additionally a subset of the files (mainly post preprocessing) are presented in the google drive folder for easy download: [here](https://drive.google.com/drive/folders/1U32mve6EUdxBUZ1EGjzxJYs5Ny91Q8vb?usp=share_link)

## Notebooks
The [notebooks](notebooks) in this repository serve as walkthroughs for the analysis performed. They include derivations of the mathematical implementations, explanations of key choices made, and present the main results. Thirteen notebooks are provided:
| Notebook | Description |
|----------|-------------|
| [Investigating M3](notebooks/1_M3_Investiagtion.ipynb) | Evaluates the effectiveness of quality and kinematic cuts on imported GAIA data by applying them to the well-studied globular cluster M3, serving as a benchmark to validate choices. |
| [Investigating NGC1851](notebooks/2_NGC1851_Investiagtion.ipynb) | Similiar notebook to that provided above however applies the selections to data around globular cluster NGC1851 to allow robust and generalisable tuning. |
| [Investigating Known Data](notebooks/3_Known_Investigation.ipynb) | Explores previously published catalogues, to understand the contextualised effects of the data cuts on the recoverable structures. |
| [Outline Preprocessing](notebooks/4_Preprocessing.ipynb) | Provides a detailed breakdown of filtering steps, parallax inversion, distance cuts, and photometric cleaning used to isolate high-quality stellar samples from raw GAIA inputs as well as how orbital parameters are calculated. |
| [Density Map 1](notebooks/5.1_DensityMap.ipynb) | Explores the application of colour scaling, false RGB binning, and various normalisation schemes to visualise the all-sky GAIA dataset. These density maps are used to highlight tidal streams and overdensities, serving as benchmarks for automated detection methods. |
| [Density Map 2](notebooks/5.2_MoreDensityMaps.ipynb) | Extends [Notebook 5.1](notebooks/5.1_DensityMap.ipynb) by removing the galactic disk using an lattitidue cut (abs(b) > 10), improving contrast and aiding the identification of fainter halo overdensities. |
| [Identifying Globuluar Clusters 1](notebooks/6.1_Glob_Clust_HigherPM.ipynb) | Presents a novel clustering technique on datatset with higher proper motion cut (ie more inclusive) to detect and aggregate recover known globular clusters, evaluating the effectiveness of proper motion cuts and density thresholds in isolating compact stellar systems. |
| [Identifying Globuluar Clusters 2](notebooks/6.2_Glob_Clust_LowerPM.ipynb) | Applies the same clustering approach as [Notebook 6.1](notebooks/6.1_Glob_Clust_HigherPM.ipynb) but with a reduced proper motion threshold, aiming to identify more distant or slower-moving globular clusters that may have been missed in the higher cut analysis. |
| [Globular Clusters with Extreme Deconvolution](notebooks/6.3_XD_Clustering.ipynb) | Presents a ready to use pipeline for Extreme Deconvolution (XD) to model measurement uncertainties in proper motion space and identify candidate globular clusters, providing a probabilistic framework that extends beyond hard thresholding methods. |
| [Identifying Tidal Streams](notebooks/7_Tidal_Stream.ipynb) | Presents an overview of key data dimensions and clustering techniques (e.g., HDBSCAN) used to detect stellar streams. The approach leverages adiabatic invariants—such as integrals of motion—to link stars not aligned in orbital phase but sharing common origins.|
| [Incorporating Chemical Data](notebooks/8_Apogee_Comparison.ipynb) | Compares Gaia-selected stellar populations with high-resolution spectroscopic data from APOGEE to investigate chemical signatures of globular clusters and assess trends in chemical evolution. |
| [Final Maps Creating](notebooks/9_Final_Map.ipynb) | Presents a comparison algorithmically identified features with visually observed overdensities in the earlier [density maps](notebooks/5.1_DensityMap.ipynb), to assess the success of the automated methods. |
| [Investigating Unidentified Overdensities](notebooks/10_Unmatched_Investigation.ipynb) | Investigates structures flagged by automated detection methods, performing cross-comparisons with extended catalogues to assess whether they correspond to known substructures, false positives, or potentially novel discoveries. |

## Results
The repository includes stored [results](data) for the pipelines intermediatary results such as tabulated identifications etc. This allows the more computationally demanding tasks to be skipped if desired as well as downstream analysis in future notebooks. Additionally in the [unknown data](data_unknown) directory, we store the GAIA samples from the region in which identify previously unmatched structures which are further investigated in [Notebook 10](notebooks/10_Unmatched_Investigation.ipynb) to determine their origin.


## Installation and Usage

To run the notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.

```bash
git clone https://github.com/JacobTutt/Galactic_Archeology.git
cd jlt67
```

### 2. Create a Fresh Virtual Environment
Use a clean virtual environment to avoid dependency conflicts.
```bash
python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows
```

### 3. Install the dependencies
Navigate to the repository’s root directory and install the package dependencies:
```bash
cd jlt67
pip install -r requirements.txt
```

### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (Galactic Archeology)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel **Python(Galactic Archeology)** to run the code.

## For Assessment
- The associated project report can be found under [Project Report](report/report.pdf). 

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE) file for details.

## Support
If you have any questions, run into issues, or just want to discuss the project, feel free to:
- Open an issue on the [GitHub Issues](https://github.com/JacobTutt/Galactic_Archeology/issues) page.  
- Reach out to me directly via [email](mailto:jacobtutt@icloud.com).

## Author
This project is maintained by Jacob Tutt 
