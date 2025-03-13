
from pyfluids import Fluid, FluidsList, Input
import matplotlib.pyplot as plt
import addcopyfighandler
import sympy as sy
import numpy as np

'''
This script is for designing a sCO2 fluid velocity
and pipe internal diameter, based on conditions 
around the turbine and an assumed pressure loss per
250m of piping. This gives a system of two equations
and two unknowns, which is solved using sympy for
a range of pressure losses, and then plotted. 

150 ft/s and a pipe diameter of ~0.45 meters result
in very low pressure losses. This is partially due
to the very low viscosity of sCO2, which results in
very high Reynold's numbers and low friciton factors. 
'''

#---Given Parameters
Ti = 700        # [C]   turbine inlet temperature
Ph = 25e6       # [Pa]  high pressure
Pl = 10e6       # [Pa]  low pressure
nt = 0.90       # [-]   turbin efficiency
qd = 148.2e6    # [W]   turbine power
Lp = 250.0      # [m]   length of pipe
dP = 0.1e6      # [Pa]  pressure drop
ep = 0.000001   # [-]   pipe roughness

#---Fluid Properties
co2_warm = Fluid(FluidsList.CarbonDioxide).with_state(
    Input.temperature(Ti), Input.pressure(Ph)
)

co2_cold = Fluid(FluidsList.CarbonDioxide).with_state(
    Input.entropy(co2_warm.entropy), Input.pressure(Pl)
)

h1a = co2_warm.enthalpy
h2s = co2_cold.enthalpy
h2a = h1a - nt * (h1a - h2s)

rh = co2_warm.density           # [kg/m3]
mu = co2_warm.dynamic_viscosity # [Pa-s]
dh = h1a - h2a                  # [J/kg]

#---Symbolic Solver
Dh_set = []
vf_set = []
dP_set = np.linspace(0.1e3, 0.1e6, 100)
for dP in dP_set: 

    Dh, vf = sy.symbols('Dh vf', reals=True, positive=True)

    Re = rh * vf * Dh / mu
    fl = 64 / Re
    ft = ((-2*sy.log((2*ep / (7.54 * Dh)) - (5.02 / Re) * sy.log((2 * ep / (7.54 * Dh)) + (13 / Re), 10), 10))**(-2)) * (1 + (Dh / Lp)**0.7)

    L1 = fl * (Lp / Dh) * (1/2) * rh * vf**2 - dP
    L2 = 2 * sy.sqrt(qd / (sy.pi * rh * vf * dh)) - Dh

    T1 = fl * (Lp / Dh) * (1/2) * rh * vf**2 - dP
    T2 = 2 * sy.sqrt(qd / (sy.pi * rh * vf * dh)) - Dh

    L_solution = sy.solve((L1, L2), (Dh, vf), dict=True)
    L_solution = L_solution[0]
    L_Re = Re.subs(L_solution)

    T_solution = sy.solve((T1, T2), (Dh, vf), dict=True)
    T_solution = T_solution[0]
    T_Re = Re.subs(T_solution)

    if L_Re <= 2300: 
        print("Oh, look! Laminar flow.")
        Dh_set.append(L_solution[Dh])
        vf_set.append(L_solution[vf])
    elif L_Re >= 2300 and T_Re >= 2300: 
        Dh_set.append(T_solution[Dh])
        vf_set.append(T_solution[vf])
    else: 
        print("Weird results, using turbulent solution.")
        print(f"L_Re = {L_Re}")
        print(f"T_Re = {T_Re}")
        Dh_set.append(T_solution[Dh])
        vf_set.append(T_solution[vf])

plt.subplot(1, 2, 1)
plt.plot(dP_set, vf_set)
plt.xlabel('Pressure Loss per 250m [Pa]')
plt.ylabel('Fluid Velocity [m/s]')
plt.title('Fluid Velocity Relationship')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(dP_set, Dh_set)
plt.xlabel('Pressure Loss per 250m [Pa]')
plt.ylabel('Hydraulic Diameter [m]')
plt.title('Hydraulic Diameter Relationship')
plt.grid()

# plt.subplot(1, 3, 3)
# plt.plot(Dh_set, vf_set)
# plt.xlabel('Hydraulic Diameter [m]')
# plt.ylabel('Fluid Velocity [m/s]')
# plt.title('Fluid Velocity vs Hydraulic Diameter')
# plt.grid()

plt.tight_layout()
plt.show()


