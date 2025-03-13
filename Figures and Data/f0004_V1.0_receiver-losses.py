'''
Kaleb Troyer
2024-08-23

Generates an approximation for receiver advective and radiative
losses by surface area and power delivered to the receiver. 

Data taken from: 
    González-Portillo, Luis & Albrecht, Kevin & Ho, Clifford. (2021). 
    Techno-Economic Optimization of CSP Plants with Free-Falling Particle 
    Receivers. Entropy. 23. 76. 10.3390/e23010076. 

Note: 2024_rec-loss_c-550.csv was fabricated through 
interpolation to improve the fit. 
'''

import os
import csv
import numpy as np
import pandas as pd
import utilities as ut
import addcopyfighandler
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

clock = ut.timer()

savefig = False
plotter = True
path = os.path.join(os.getcwd(), 'Figures and Data')
f375 = '2024-08-21_rec-loss_b-375.csv'
f750 = '2024-08-21_rec-loss_a-750.csv'
f550 = '2024-08-21_rec-loss_c-550.csv'

ax, az, ay = [], [], []
bx, bz, by = [], [], []
cx, cz, cy = [], [], []

with open(os.path.join(path, 'data', f375), 'r') as file: 
    reader = csv.reader(file)
    for row in reader: 
        by.append(375)
        bx.append(float(row[0]))
        bz.append(float(row[1]))

with open(os.path.join(path, 'data', f750), 'r') as file: 
    reader = csv.reader(file)
    for row in reader: 
        ay.append(750)
        ax.append(float(row[0]))
        az.append(float(row[1]))

with open(os.path.join(path, 'data', f550), 'r') as file: 
    reader = csv.reader(file)
    for row in reader: 
        cy.append(550)
        cx.append(float(row[0]))
        cz.append(float(row[1]))

def polynomial(XY, a0, a1, a2, a3): 
    x, y = XY
    return sum([
                a0, 
                a1*x**1, 
                a2*x**2, 
                a3*x*y, 
            ])

area = ax + bx + cx
MWth = ay + by + cy
loss = az + bz + cz

XY = np.vstack((area, MWth))
popt, pcov = curve_fit(polynomial, XY, loss)

# fit results
Zfit = polynomial(XY, *popt)
rmse = np.sqrt(np.mean((loss - Zfit)**2))
print(f"RMSE {rmse:5.3f}")
for coeff in range(len(popt)):
    print(f"a{coeff:<1} = {popt[coeff]:>10.3e}")

# getting fit-plot ready
x = np.linspace(100, 1500, 30)
y = np.linspace(20, 1500, 30)
xt, yt = np.meshgrid(x, y)
xy = np.vstack([xt.ravel(), yt.ravel()])
z = polynomial(xy, *popt).reshape(xt.shape)
z[z < 0] = np.nan

# 3d plot of data and fit
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xt, yt, z, alpha=0.5)
ax.scatter(area, MWth, loss, color='red', label='(González-Portillo, 2021)')
ax.set_xlabel(r'Curtain Area [$m^{2}$]')
ax.set_zlabel(r'Thermal Loss / Input Power [-]')
ax.set_ylabel(r'Input Power [MWt]')
ax.legend()

if savefig: fig.savefig(os.path.join(path, 'figures', 'receiver-losses.png'), dpi=300, bbox_inches='tight')
if plotter: plt.show()
