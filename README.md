# Galactic Archeology Coursework Repository

This repository contains the report and supporting code for the Galactic Archeology (A4) coursework.

- This majority of the analysis is provided in notebooks which have been seperated and numbered for progression throughout the tasks.

- Some of the large scale plotting and preprocessing code has been implemented as functions (documented with docstrings) in the `Analaysis` folder

- Information regarding the data used during this project (and the neccessary ADQL queries) are outlines in `0_Data.md`. 

---

## Installation Instructions

To run the notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a4_coursework/jlt67.git
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
python -m ipykernel install --user --name=env --display-name "Python (GA/A4 Coursework)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel **(Python(Galactic Archeology Coursework))** to run the code.



## For Assessment

### Report
Please find the projects report under `Report` directory

### Declaration of Use of Autogeneration Tools
This report made use of Large Language Models (LLMs) to assist in the development of the project.
These tools have been employed for:
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Helping with Latex formatting for the report.
- Identifying grammar and punctuation inconsistencies within the report.