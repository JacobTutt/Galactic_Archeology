# Galactic Archeology Project

This project:
- Uses Gaia to construct an all-sky Gaia map of the Milky Way's halo substructures.
- Identifies overdensities (Globuluar Glusters and satellite galaxies) using automated detection
- Detects stellar streams using Integral of Motion space
- Includes metallicity comparisons with APOGEE DR17 to understand stellar evolution.
---

## Installation Instructions

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
Open the notebooks and select the created kernel **(Python(Galactic Archeology))** to run the code.

### Report
A report for this project can be found under the Report directory of the repository
