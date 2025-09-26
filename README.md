# Numerical Coherent Mode Decomposition: Validation and Applications

This repository contains the Python scripts used to generate the figures and tables for the manuscript "Numerical Coherent Mode Decomposition: Validation and Applications" submitted to Physical Review A.

## Code Description

The scripts are organized to reproduce the main results of the paper.

*   `partcohr3.py`: This script performs the core validation simulations and generates the data and plots for **Figures 1 through 5**.

*   `tophat.py`: This script runs the "Trojan Horse" beam demonstration and generates the multi-panel plot for **Figure 6**.

*   `Delta.py`: This script runs the M² validation by varying the beam complexity (number of modes) on a fixed grid. It prints a formatted table to the console with the data for **Table I**.

*   `Charlie.py`: This script runs the M² convergence test by varying the grid resolution for the full 20-mode beam. It prints a formatted table to the console with the data for **Table II**.

## How to Run

The scripts can be run directly using a standard Python environment. The required libraries are common in scientific computing:

*   NumPy
*   SciPy
*   Matplotlib

To reproduce the figures and generate the data for the tables, execute the scripts from your terminal:

```bash
# To generate Figures 1-5
python partcohr3.py

# To generate Figure 6
python tophat.py

# To generate data for Table I
python Delta.py

# To generate data for Table II
python Charlie.py
