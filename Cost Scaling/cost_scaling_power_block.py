
import matplotlib.pyplot as plt
import addcopyfighandler
import numpy as np

'''
For viewing the cost-temperature relationship of a
turbine and a recuperator, for a given design power
and conductance, respectively. 
'''

def factor(c, d, dT): 
    ft = 1 + c * dT + d * dT**2
    return ft
def recuperator(T, UA): 
    '''
    UA [W/k]
    '''
    cost = 49.45e-6 * (UA**0.7544)

    c = 0.02141
    d = 0.0
    t = 550

    dT = T - t
    ft = factor(c, d, dT)
    return ft * cost # [M$]
def turbine(T, Wd):
    '''
    Wd [MW]
    '''
    cost = 182600e-6 * (Wd**0.5561)

    c = 0.0
    d = 1.106e-4
    t = 550

    dT = T - t
    ft = factor(c, d, dT)
    return ft * cost # [M$]

Wd = 150    # [MW]
UA = 10e6   # [W/K]

temps = np.linspace(550, 1000, 100)
tcost = turbine(temps, Wd)
rcost = recuperator(temps, UA)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

axes[0].plot(temps, tcost)
axes[0].margins(x=0)
axes[0].set_xlabel('Temperature [C]')
axes[0].set_ylabel('Capital [M$]')
axes[0].set_title(r'Turbine Cost ($\dot{W}$ = '+f'{Wd:.1f} [MW])')
axes[0].grid()

axes[1].plot(temps, rcost)
axes[1].margins(x=0)
axes[1].set_xlabel('Temperature [C]')
axes[1].set_title(f'Recuperator Cost (UA = {UA * 1e-6:.1f} [MW/K])')
axes[1].tick_params(axis='y', left=False)
axes[1].grid()

plt.tight_layout()
plt.show()


