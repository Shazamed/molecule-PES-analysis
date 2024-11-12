import numpy as np
import matplotlib.pyplot as plt
import os 
import re
from scipy.optimize import curve_fit
import natsort
from matplotlib import cm


# conversion
m_u = 1.660539068e-27

molecule = "H2S"

h2o_directory_path = "./part2programming/Ex2/H2Ooutfiles/"
h2s_directory_path = "./part2programming/Ex2/H2Soutfiles/"
angle_length_re = r'H\s+1\s+\d+.\d*\s+2\s+\d+.'
energy_re = r'E\(RHF\)\s*=\s*-?\d+.\d*'

h2o_files_list = natsort.natsorted(os.listdir(h2o_directory_path))
h2s_files_list = natsort.natsorted(os.listdir(h2s_directory_path))

def vib_model(data, kr, kf):
    r, theta = data
    return 0.5 * kr * r**2 + 0.5 * kf * theta**2

def extract_values(directory_path, files_list):
    values_array = np.empty((0,3))
    for out_file in files_list: # iterate over the files
        with open(directory_path + out_file, 'r') as f: # regex and consolidating data
            f_text = f.read()
            re_match = re.search(angle_length_re, f_text).group()
            bond_angle = float(re_match.split()[-1])
            bond_length = float(re_match.split()[2])

            re_match = re.search(energy_re, f_text).group()
            bond_energy= float(re_match.split()[2])

            values_array = np.vstack([values_array, [bond_length, bond_angle, bond_energy]])
    return values_array

def find_equilibrium(xx,yy,zz):
    e_min = zz.min() # energy minimum
    min_coord = np.where(zz==e_min) # find coordinates of minimum
    min_coord = [min_coord[0][0], min_coord[1][0]] # convert to list [x,y]
    bond_length_eq = xx[min_coord[0],min_coord[1]] # get bond length of eq from coords
    bond_angle_eq = yy[min_coord[0],min_coord[1]] # get bond angle of eq from coords
    print(f"Equilibrium bond length: {bond_length_eq} Å")
    print(f"Equilibrium bond angle: {bond_angle_eq}°")
    print(f"Equilibrium energy: {e_min} Eh")
    return (bond_length_eq,bond_angle_eq,e_min)


if molecule.upper() == "H2O":
    molecule_array = extract_values(h2o_directory_path, h2o_files_list)
elif molecule.upper() == "H2S":
    molecule_array = extract_values(h2s_directory_path, h2s_files_list)
else:
    print("Molecule is wrong")
    quit()

X = molecule_array[:,0] # bond length array
Y = molecule_array[:,1] # bond angle array
Z = molecule_array[:,2] # energy array

xx, yy = np.meshgrid(np.unique(molecule_array[:,0]),np.unique(molecule_array[:,1])) # generate mesh grid of the bond angle and length
zz = Z.reshape(len(np.unique(molecule_array[:,0])),len(np.unique(molecule_array[:,1]))).T

bond_length_eq, bond_angle_eq, e_min = find_equilibrium(xx,yy,zz)

well_coords = np.where(zz<=e_min+0.025) # set energy threshold above well to select points
e_well  = zz[*well_coords]
vib_r_array = xx[*well_coords] - bond_length_eq
vib_theta_array = yy[*well_coords] - bond_angle_eq

popt, pcov = curve_fit(vib_model, (vib_r_array*1e-10,vib_theta_array*np.pi/180), (e_well-e_min)*43.60e-19)

z_fit = vib_model((vib_r_array*1e-10,vib_theta_array*np.pi/180), *popt)/43.60e-19+e_min

vib_freq_1 = (1/(2*np.pi) * (popt[0]/(2*m_u))**0.5)
vib_freq_2 = (1/(2*np.pi) * (popt[1]/(0.5*m_u*(bond_length_eq*1e-10)**2))**0.5)


print(f'Stretching frequency, ν_1 = {vib_freq_1*3.33565e-11:.2f} cm^-1')
print(f'Bending frequency, ν_2 = {vib_freq_2*3.33565e-11:.2f} cm^-1')


# plotting
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
if molecule.upper() == "H2O":
    surface = ax1.plot_surface(xx, yy, zz, cmap=cm.plasma , linewidth=0.5, rstride=6, cstride=1, edgecolors="k",alpha=.8,vmin=e_min)
else:
    surface = ax1.plot_surface(xx[:,6:], yy[:,6:], zz[:,6:], cmap=cm.plasma , linewidth=0.5, rstride=6, cstride=1, edgecolors="k",alpha=.8,vmin=e_min)
fig1.colorbar(surface, shrink=0.5, aspect=5)
ax1.set_xlabel("Bond length/Å")
ax1.set_ylabel("Bond angle/°")
ax1.set_zlabel("Energy/Eh")
ax1.set_title(f"Potential Energy Surface of {molecule}")
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_trisurf(vib_r_array+bond_length_eq,vib_theta_array+bond_angle_eq,e_well, label="Data", alpha=0.5)
ax2.plot_trisurf(vib_r_array+bond_length_eq,vib_theta_array+bond_angle_eq,z_fit, label="Fit", alpha=0.5)
ax2.set_xlabel("Bond length/Å")
ax2.set_ylabel("Bond angle/°")
ax2.set_zlabel("Energy/Eh")
ax2.legend()
ax2.set_title("Comparison between Selected Data and Fitted Energies")

fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')
if molecule.upper() == "H2O":
    surface = ax3.plot_surface(xx, yy, zz, cmap=cm.plasma , linewidth=0.5, rstride=6, cstride=1, edgecolors="k",alpha=.5,vmin=e_min)
else:
    surface = ax3.plot_surface(xx[:,6:], yy[:,6:], zz[:,6:], cmap=cm.plasma , linewidth=0.5, rstride=6, cstride=1, edgecolors="k",alpha=.5,vmin=e_min)
fig3.colorbar(surface, shrink=0.5, aspect=5)
ax3.set_xlabel("Bond length/Å")
ax3.set_ylabel("Bond angle/°")
ax3.set_zlabel("Energy/Eh")
ax3.plot_trisurf(vib_r_array+bond_length_eq,vib_theta_array+bond_angle_eq,e_well, label="Data", alpha=1, color='red')
ax3.set_title(f"Potential Energy Surface of {molecule} with Selected Data")
plt.show()
