Examples

This folder contains four example systems you can use to test the code and see how it runs and what files it produces.

Each example trajectory has only ~20 frames — not enough for a reliable fit. For the full trajectories, see the Zenodo archive:

DOI: https://doi.org/10.5281/zenodo.23020457

Requirements

Before running an example, install the code's dependencies — see the main README in the LipiDeA_v1.0/ directory for the full list and setup instructions.

Running an example

From any of the example folders, call main.py and pass the input file for the system you want to run:

python path/to/code/main.py LD.in
python path/to/code/main.py DPPC_bilayer.in

Replace path/to/code/main.py with the actual path to main.py on your machine.

Each .in file is a self-contained configuration for one example system — edit a copy of it if you want to change parameters, rather than modifying the original.
