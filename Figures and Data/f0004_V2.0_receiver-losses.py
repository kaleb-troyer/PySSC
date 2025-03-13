'''
Kaleb Troyer
2024-11-17

Generates an approximation for receiver advective and radiative
losses by surface area and power delivered to the receiver. 

Developed using loss correlations from Sandia (https://www.osti.gov/biblio/1890267, page 43)

Expanded from the same experimentation as f0004_V1.0. Considers wind velocity and is 
generally more suitable for the whole design space. 
'''

import numpy as np
import addcopyfighandler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

figID   = 'f0004'
version = '2.0'
savefig = False
display = True

def air_velocity(Ht, Hr, v): 
    E1 = np.log((Ht + Hr / 2) / 0.003)
    E2 = np.log(10.0 / 0.003)
    return (E1 / E2) * v
def wind_direction(th): 
    F = 5.50
    G = 7.50
    H = 5000

    E1 = (180 - np.abs(180 - th))
    E2 = np.exp(-E1 / G) / H
    return E2 * E1**F
def eta_receiver(Qi, Ar, th, v): 
    A = 0.848109
    B = 0.249759
    C = -1.0115660
    D = -7.942869e-5
    E = -1.4575091E-07

    qs = np.exp(-Qi / Ar)
    TH = wind_direction(th)
    vm = air_velocity(400, 20, v)

    E1 = A
    E2 = B * qs
    E3 = C * qs**2
    E4 = D * qs * vm * TH
    E5 = E * TH * vm**2
    return -(E1 + E2 + E3 + E4 + E5 - 1)

#---study parameters
v = 10
th = 0
Ar = np.linspace(1, 1600, 40)  # Avoid division by zero
Qi = np.linspace(1, 1400, 40)  # Avoid division by zero

# generating the mesh and plotting
Ar_mesh, Qi_mesh = np.meshgrid(Ar, Qi)
eta_mesh = eta_receiver(Qi_mesh, Ar_mesh, th, v)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(Ar_mesh, Qi_mesh, eta_mesh, cmap='viridis', alpha=0.5)

# building the figure
ax.set_xlabel(r"Receiver Area [m$^{2}$]")
ax.set_ylabel("Input Power [MWt]")
ax.set_zlabel("Thermal Loss / Input Power [-]")

path = os.path.join(os.getcwd(), "Figures and Data")
if savefig: plt.savefig(os.path.join(path, "figures", f"{figID}_V{version}_receiver-losses.png"), dpi=300, bbox_inches='tight')
if display: plt.show()

