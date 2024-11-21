# molecule-PES-analysis
Analyse and plot the Potential Energy Surface (PES) of $H_2O$ and $H_2S$.
## Description
The script extracts data from a set of gaussian output files in the part2programming/Ex2/[molecule]outfiles folder and plots out the Potential Energy Surface. 

The script will determine the equilibrium bond angle, length and energy of the molecule from the given set of data. The energies are fitted and used to determined the vibrational frequencies of the stretching and bending mode of the molecule.
## Usage
Navigate to the directory where the main.py and part2programming folder are both located

Then to run just simply run main.py with python:
```
python main.py
```