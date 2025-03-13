
from pyfluids import Fluid, FluidsList, Input
import matplotlib.pyplot as plt
from PyHXsim import Sand
import sympy as sy
import numpy as np

TIT = 700
PIT = 940

Twarm = (655.14, 940.0)
Tcold = (535.14, 700.0)
posit = (0, 1)

plt.plot(posit, Twarm, color='red', label='Sand')
plt.plot(posit, Tcold, color='blue', label=r'sCO$_{2}$')
plt.scatter(posit, Twarm, color='red', zorder=3)
plt.scatter(posit, Tcold, color='blue', zorder=3)
plt.plot([posit[0], posit[0]], [Tcold[0], Twarm[0]], color='gray', linestyle='--') # Cold Approach Temperature
plt.plot([posit[1], posit[1]], [Tcold[1], Twarm[1]], color='gray', linestyle='--') # Hot Approach Temperature

# Add arrows indicating direction of flow
mid_x = (posit[0] + posit[1]) / 2  # Midpoint in x-direction
mid_y_warm = (Twarm[0] + Twarm[1]) / 2  # Midpoint of warm line
mid_y_cold = (Tcold[0] + Tcold[1]) / 2  # Midpoint of cold line

slope_warm = (Twarm[1] - Twarm[0]) / (posit[1] - posit[0]) 
slope_cold = (Tcold[1] - Tcold[0]) / (posit[1] - posit[0]) 
angle_warm = np.degrees(np.arctan(slope_warm)) 
angle_cold = np.degrees(np.arctan(slope_cold)) 

dx = 0.1  # Small step in x-direction
dy_warm = slope_warm * dx
dy_cold = slope_cold * dx

plt.annotate(
    "", xy=(mid_x - dx, mid_y_warm - dy_warm), xytext=(mid_x + dx, mid_y_warm + dy_warm),
    arrowprops=dict(arrowstyle="->", color="red", linewidth=1.0)
)  # Red arrow pointing left along slope

plt.annotate(
    "", xy=(mid_x + dx, mid_y_cold + dy_cold), xytext=(mid_x - dx, mid_y_cold - dy_cold),
    arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.0)
)  # Blue arrow pointing right along slope

# Add labels for TIT and PIT
plt.text(posit[1], TIT, 'Turbine Inlet  \nTemperature (TIT)  ', verticalalignment='bottom', horizontalalignment='right', fontsize=10, color='black')
plt.text(posit[1], PIT, 'PHX Inlet  \nTemperature (PIT)  ', verticalalignment='bottom', horizontalalignment='right', fontsize=10, color='black')

# Add labels for approach temperatures
plt.text(
    posit[0], 25+(Twarm[0] + Tcold[0]) / 2, '  Cold Approach\n  Temperature'+r' $(dT_{c})$', 
    verticalalignment='center', horizontalalignment='left', fontsize=10, color='gray'
)
plt.text(
    posit[1], -20+(Twarm[1] + Tcold[1]) / 2, 'Hot Approach  \nTemperature'+r' $(dT_{h})$  ', 
    verticalalignment='center', horizontalalignment='right', fontsize=10, color='gray'
)
plt.text(
    mid_x-0.07, 5+mid_y_warm, r'$\dot{m}_{sand}$', 
    verticalalignment='center', horizontalalignment='right', fontsize=10, color='red'
)
plt.text(
    mid_x+0.15, mid_y_cold-10, r'$\dot{m}_{sCO_{2}}$', 
    verticalalignment='center', horizontalalignment='right', fontsize=10, color='blue'
)

plt.legend()
plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.xlabel('Position in PHX')
plt.ylabel('Fluid Temperature')
plt.ylim(top=1000)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('PHX profile.png', dpi=300)
plt.show()


