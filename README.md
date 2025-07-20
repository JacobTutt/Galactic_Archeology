# 

## Author: Jacob Tutt, Department of Physics, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Descriptions

This project:
- Uses Gaia to construct an all-sky Gaia map of the Milky Way's halo substructures.
- Identifies overdensities (Globuluar Glusters and satellite galaxies) using automated detection
- Detects stellar streams using Integral of Motion space
- Includes metallicity comparisons with APOGEE DR17 to understand stellar evolution.

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
During this project, data was aquired from the Gaia ESA Archive using ADQL queries. To allow for reproducibility, the [data file](0_Data.md) file provides an outline of the raw files used as well as the motivation behind them and the necessary queries to recreate them. 

Additionally a subset of the files (mainly post preprocessing) in this are presented in the google drive folder: [here](https://drive.google.com/drive/folders/1U32mve6EUdxBUZ1EGjzxJYs5Ny91Q8vb?usp=share_link)

## Notebooks
The [notebooks](notebooks) in this repository serve as walkthroughs for the analysis performed. They include derivations of the mathematical implementations, explanations of key choices made, and present the main results. Three notebooks are provided:
| Notebook | Description |
|----------|-------------|



## Results


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
