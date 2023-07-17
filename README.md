[![Python 3.10.6](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3106/)
# Tumor Growth
Implementation of a mammary tumor growth ODE model in rats (PA from Computational Modeling course - DCOMP - UFSJ).  

# Requirements

- [Python3](https://python.org) and [pip](https://pip.pypa.io/en/stable/installation/) package manager:

      sudo apt install python3 python3-pip build-essential python3-dev

- [pandas](https://pandas.pydata.org/) library:

      pip install pandas

- [SciPy](https://scipy.org/) library:

      pip install scipy
       
- [Matplotlib](https://matplotlib.org/) library:
 
      pip install matplotlib

- [numpy](https://numpy.org/) library:

      pip install numpy

## To install all dependencies

    ./install_dependencies.sh

# Execution

You can change the referred dataset (treated or untreated rats), plot the error evolution for each one of them, alter cisplatin dose, as well as raise or lower the ODE delay, just by modifying some source code lines.

## Parameter Estimation

    python3 differential_evolution.py
    
## Simulation
      
    python3 simulate.py
