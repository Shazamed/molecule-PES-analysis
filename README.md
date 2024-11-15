# molecule-PES-analysis
Analyse and plot the Potential Energy Surface (PES) of $H_2O$ and $H_2S$.
## Description
The script extracts data from a set of gaussian output files in the part2programming/Ex2/[molecule]outfiles folder and plots out the Potential Energy Surface. 

The script will determine the equilibrium bond angle, length and energy of the molecule from the given set of data. The energies are fitted and used to determined the vibrational frequencies of the stretching and bending mode of the molecule.
## Usage
To run just simply run main.py with python:
```
python main.py
```
To change molecule, edit the following line in the code and change the string to either "H2S" or "H2O":
```python
molecule = "H2S"
```