# Starbucks promotion type exercise

### Summary
This project is a recommendation exercise provided by Starbucks.
The aim is to investigate customer receptivity to different promotions for a product.
The setup is quite simple.
It includes data on customers with their details and information on whether a promotion was sent or not, the type of promotion, monetary transactions and whether the offer was subsequently viewed or completed.
A full description of the setup can be found in the main Jupyter file.
A documentation of the introduction, technical details and implementation as well as results is provided in the file "report.pdf".

### Motivation
This project deals with a simple real-life problem of shops, restaurants or cafés: how to attract the right customers who are willing to pay for the products offered.
From its peculiarities, one can learn to improve intuition in marketing campaigns.

### Data sources 
The following Json files constitute as data source:
- portfolio.json                       # enlists offer data
- profile.json                         # enlists demographic data
- transcript.json                      # enlists transaction data

### Acknowledgements 
The used data stems from the company "Starbucks" and represents data of a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products. The data was provided through the company "udacity" in the framework of a data science online course.

### Install/run instructions:
1. Install python >= 3.11 and the packages denoted in requirements_working_configuration.txt preferably in a virtual environment as exemplarily shown for the windows command prompt:
```
      projectfolder:> python -m venv .venv
      projectfolder:> cd .venv\Scripts
      projectfolder\.venv\Scripts> .\activate.bat
      projectfolder\.venv\Scripts> cd ..\..
      projectfolder:> python -m pip install -r requirements_working_configuration.txt
```
      Within your project folder "projectfolder" use the "python" command as long as the virtual environment is activated
      (if not working with/on the project, the virtual environment should be deactivated by executing projectfolder\.venv\Scripts\deactivate.bat).

      Important note: If you are using requirements.txt (no package versions provided) instead of requirements_working_configuration.txt, make shure to install ipykernel before (python -m pip install ipykernel). The order seems important.

2. In the programming IDE select .venv\Scripts\python.exe as Kernel.

1. Run all code blocks in "Starbucks.ipynb"

### Files in the repository
```
.gitignore
classifier_module.py                   # python module used in main Jupyter file
data
|-portfolio.json                       # enlists offer data
|-profile.json                         # enlists demographic data
|-transcript.json                      # enlists transaction data
dataflow.pptx                          # pic file in documentation
dataflow.png                           # pic file in documentation
pics
|-*.png                                # picture files used in the documentation
pic1.png                               # pic file included in main jupyter file
pic2.png                               # pic file included in main jupyter file
README.md
report.aux                             # documentation file
report.log                             # documentation file
report.out                             # documentation file
report.pdf                             # documentation file
report.synctex.gz                      # file enabling synchronization between source document and the PDF output
report.tex                             # documentation source file
requirements.txt                       # list of required packages
requirements_working_configuration.txt # list of packages covering a working configuration
results
|-*.png                                # pic files in the results part of the documentation
Starbucks_Capstone_notebook.ipynb      # main jupyter file
venn_diagram2.png                      # pic file used in documentation
```
