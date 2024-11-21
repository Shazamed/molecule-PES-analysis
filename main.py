import numpy as np
import matplotlib.pyplot as plt
import os 
import re
from scipy.optimize import curve_fit
import natsort
from matplotlib import cm

# conversion
m_u = 1.660539068e-27 # in kg
angstrom = 1e-10 # in kg
hartree_to_joule = 43.60e-19 # convert hartree to joule
speed_of_light_cm = 2.998e10 # speed of light in cm/s

UPPER_E = 0.025 # upper bound of energies above the minima used for fitting

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
    '''Function uses regex to search for expressions that matches the regex string across the files.
    Regex extracts bond angle, bond length and energy from the files and stores it into values_array
    values_array has the form:
    bond length/Å bond angle/°      Energy/Eh
   [[   0.6          70.         -395.79755828]
    [   0.6          71.         -395.80114662]
    [   0.6          72.         -395.80462758]
    ...
    [   1.8         158.         -398.47169166]
    [   1.8         159.         -398.46935484]
    [   1.8         160.         -398.46699712]]
    '''
    values_array = np.empty((0,3))
    print("Loading files")
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

def gen_meshgrid(X,Y,Z):
    '''Generates a 2D meshgrid for the matplotlib plot_surface() function
    xx (bond length) =
    [[0.7  0.75 0.8  ... 1.8  1.85 1.9 ] 
    [0.7  0.75 0.8  ... 1.8  1.85 1.9 ]
    ...
    [0.7  0.75 0.8  ... 1.8  1.85 1.9 ]
    [0.7  0.75 0.8  ... 1.8  1.85 1.9 ]]
    
    yy (bond angle) =
    [[ 70.  70.  70. ...  70.  70.  70.]
    [ 71.  71.  71. ...  71.  71.  71.]
    ...
    [159. 159. 159. ... 159. 159. 159.]
    [160. 160. 160. ... 160. 160. 160.]]

    zz is a 2D array of energies with the corresponding xx (bond length) and yy (bond angle) at that position:
    zz =
    [[-75.70705381 -75.82565047 ... -75.61899215]
    [-75.71098123 -75.82920916 ...   -75.6174709]
    ...
    [-75.75649832 -75.86234643 ...  -75.55837478]
    [-75.75583763 -75.86163664 ...  -75.55741546]]
    '''
    xx, yy = np.meshgrid(np.unique(X),np.unique(Y)) # generate mesh grid of the bond angle and length
    zz = Z.reshape(len(np.unique(X)),len(np.unique(Y))).T # reshape 1D array into a 2D array with same dimensions as the meshgrid
    return xx, yy, zz

def find_equilibrium(xx,yy,zz):
    '''outputs equilibrium bond length, angle and energy'''
    e_min = zz.min() # energy minimum
    min_coord = np.where(zz==e_min) # find coordinates of minimum
    x_coord, y_coord = min_coord[0][0], min_coord[1][0] # get x and y coords of min
    bond_length_eq = xx[x_coord, y_coord] # get bond length of eq from coords
    bond_angle_eq = yy[ x_coord, y_coord] # get bond angle of eq from coords
    print(f"Equilibrium bond length: {bond_length_eq} Å")
    print(f"Equilibrium bond angle: {bond_angle_eq}°")
    print(f"Equilibrium energy: {e_min} Eh")
    return (bond_length_eq,bond_angle_eq,e_min)

def gen_fit_values(xx,yy,zz, r_eq, theta_eq, e_o):
    '''Return arrays of energies, bond lengths and bond angles to be used in the fitting
    '''
    well_coords = np.where(zz<=e_o+UPPER_E) # set energy threshold above well to select points
    e_array  = zz[*well_coords]
    vib_r_array = xx[*well_coords] - r_eq # gives r - r_e
    vib_theta_array = yy[*well_coords] - theta_eq # gives theta - theta_e
    return e_array, vib_r_array, vib_theta_array

def calc_freq(k_r, k_theta, r_eq):
    '''Returns v_1 and v_2 in Hz'''
    v_1 = (1/(2*np.pi) * (k_r/(2*m_u))**0.5) 
    v_2 = (1/(2*np.pi) * (k_theta/(0.5*m_u*(r_eq*angstrom)**2))**0.5)
    return v_1, v_2


while True: # keep asking user for valid input
    print("Type 'H2O' or 'H2S' to select the molecule for analysis")
    molecule = input().upper()
    if molecule == "H2O":
        molecule_array = extract_values(h2o_directory_path, h2o_files_list)
        break
    elif molecule == "H2S":
        molecule_array = extract_values(h2s_directory_path, h2s_files_list)
        break
    else:
        print("Molecule is invalid")

X = molecule_array[:,0] # gives array of bond lengths
Y = molecule_array[:,1] # gives array of bond angles
Z = molecule_array[:,2] # gives array of energies

xx, yy, zz = gen_meshgrid(X,Y,Z) # meshgrid of bond length (xx), bond angles (yy) that has corresponding values of energies (zz) at a given coord.

bond_length_eq, bond_angle_eq, e_min = find_equilibrium(xx,yy,zz)
e_well, vib_r_well, vib_theta_well = gen_fit_values(xx,yy,zz,bond_length_eq, bond_angle_eq, e_min) # values of vib_r in angstrom, vib_theta in degrees and e_min in hartree
popt, pcov = curve_fit(vib_model, (vib_r_well*angstrom,vib_theta_well*np.pi/180), (e_well-e_min)*hartree_to_joule) # popt contains k_r in N/m and k_theta in N m/rad
vib_freq_1, vib_freq_2 = calc_freq(popt[0], popt[1], bond_length_eq) # vib_freq_1 and vib_freq_2 in Hz

print(f'Stretching frequency, ν_1 = {vib_freq_1/speed_of_light_cm:.2f} cm^-1')
print(f'Bending frequency, ν_2 = {vib_freq_2/speed_of_light_cm:.2f} cm^-1')


# plotting surface plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
if molecule.upper() == "H2O":
    surface = ax1.plot_surface(xx, yy, zz, cmap=cm.plasma , linewidth=0.5, rstride=6, cstride=1, edgecolors="k",alpha=.8,vmin=e_min)
else:
    # for H2S, plot from r = 0.9 onwards, ie. (xx[:,6:]), in order to make well more visible
    surface = ax1.plot_surface(xx[:,6:], yy[:,6:], zz[:,6:], cmap=cm.plasma , linewidth=0.5, rstride=6, cstride=1, edgecolors="k",alpha=.8,vmin=e_min)

fig1.colorbar(surface, shrink=0.5, aspect=5)
ax1.set_xlabel("Bond length/Å")
ax1.set_ylabel("Bond angle/°")
ax1.set_zlabel("Energy/Eh")
ax1.set_title(f"Potential Energy Surface of {molecule}")
plt.show()

# show comparison between fit and data
z_fit = vib_model((vib_r_well*angstrom,vib_theta_well*np.pi/180), *popt)/hartree_to_joule+e_min

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_trisurf(vib_r_well+bond_length_eq,vib_theta_well+bond_angle_eq,e_well, label="Data", alpha=0.5)
ax2.plot_trisurf(vib_r_well+bond_length_eq,vib_theta_well+bond_angle_eq,z_fit, label="Fit", alpha=0.5)
ax2.set_xlabel("Bond length/Å")
ax2.set_ylabel("Bond angle/°")
ax2.set_zlabel("Energy/Eh")
ax2.legend()
ax2.set_title("Comparison between Selected Data and Fitted Energies")

# show selected points on surface plot
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
ax3.plot_trisurf(vib_r_well+bond_length_eq,vib_theta_well+bond_angle_eq,e_well, label="Data", alpha=1, color='red')
ax3.set_title(f"Potential Energy Surface of {molecule} with data used for fitting")
plt.show()