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
from utilities import colorGenerator
import os

figID   = 'f0007'
version = '1.0'
savefig = False
display = True

# Receiver efficiency correlations from Sandia (https://www.osti.gov/biblio/1890267, page 43)
def air_velocity(Ht, Hr, v): 
    E1 = np.log((Ht + Hr / 2) / 0.003)
    E2 = np.log(10.0 / 0.003)
    return (E1 / E2) * v
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
    return E1 + E2 + E3 + E4 + E5
def wind_direction(th): 
    F = 5.50
    G = 7.50
    H = 5000

    E1 = (180 - np.abs(180 - th))
    E2 = np.exp(-E1 / G) / H
    return E2 * E1**F

# plotting all wind velocities
colors = colorGenerator(factor=0.1)
angles = np.linspace(0, 180, 300)
for v in np.linspace(0, 10, 11): 
    scales = eta_receiver(200, 200, angles, v)
    plt.plot(angles, scales, color=next(colors), label=r"v$_{g}$"+f" = {v:3.1f} [m/s]")

# building the figure
plt.grid()
plt.margins(x=0)
plt.legend()
plt.title(r'FFPR Efficiency for 200MWt Input and 200m$^{2}$ Aperature')
plt.xlabel('Wind Direction (North=0deg for a North-Facing Receiver)')
plt.ylabel('Receiver Thermal Efficiency')

path = os.path.join(os.getcwd(), "Figures and Data")
if savefig: plt.savefig(os.path.join(path, "figures", f"{figID}_V{version}_windspeed-loss-evaluation.png"), dpi=300, bbox_inches='tight')
if display: plt.show()


