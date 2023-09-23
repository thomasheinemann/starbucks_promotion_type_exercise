# Starbucks promotion type exercise

### Summary
This project is a recommendation exercise provided by Starbucks.
The goal is to investigate customers receptiveness towards various promotions for a product.
The setup is quite simplified.
It covers data of customers with their specifics and the information on whether a promotion was sent or not, the type of promotion and if the product was bought afterwards.
A full description of the problem setting is provided in the main jupyter file.

### Motivation
This project covers a simple real life problem of shops, restaurants, or cafés in how to advertise he right customers that are willing to pay for offered products.
From their specifics one can learn improve intuition in the marketing campaigns.

### Data sources: 

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
classifier_module.py                   # 
data
|-portfolio.json                       # enlists offer data
|-profile.json                         # enlists demographic data
|-transcript.json                      # enlists transaction data
dataflow.pptx                          # pic file in documentation
dataflow.png                           # pic file in documentation
pic1.png                               # pic file included in main jupyter file
pic2.png                               # pic file included in main jupyter file
README.md
report.pdf                             # documentation file
report.tex                             # documentation source file
requirements.txt                       # list of required packages
requirements_working_configuration.txt # list of packages covering a working configuration
results
|-*.png                                # pic files in documentation
Starbucks_Capstone_notebook.ipynb      # main jupyter file
venn_diagram.png                       # pic file used in documentation
venn_diagram2.png                      # pic file used in documentation
```
