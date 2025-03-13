import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import state_finder, table_builder
from scipy.optimize import minimize

from pyfluids import Fluid, FluidsList, Input

# HTF = Fluid(FluidsList.CarbonDioxide)

# pressure = 7.5e6
# temperature_range = (300, 350)

# specific_heats = []

# def find_Cp(df, temperature):
#     # Find the index for interpolation
#     idx = (df['temp.'] - temperature).abs().idxmin()
#     if idx == 0:
#         return df.loc[idx, 'Cp']  # Handle edge case for first row
#     elif idx == len(df) - 1:
#         return df.loc[idx, 'Cp']  # Handle edge case for last row
#     else:
#         # Interpolate Cp value for the closest matching temperature
#         prev_temp = df.loc[idx - 1, 'temp.']
#         next_temp = df.loc[idx + 1, 'temp.']
#         prev_cp = df.loc[idx - 1, 'Cp']
#         next_cp = df.loc[idx + 1, 'Cp']
#         interpolated_cp = prev_cp + ((temperature - prev_temp) / (next_temp - prev_temp)) * (next_cp - prev_cp)
#         return interpolated_cp

# table = table_builder(pressure)
# print(table)

# for temp in np.arange(temperature_range[0], temperature_range[1], 1):
#     HTF.update(Input.pressure(pressure), Input.temperature(temp))
#     Cp_calc = HTF.as_dict()['specific_heat']/1000
#     Cp_interp = find_Cp(table, temp)
#     # print(Cp_interp)

#     specific_heats.append({'Temperature': temp, 'Cp interpolated': Cp_interp, 'Cp calculated': Cp_calc})

# df = pd.DataFrame(specific_heats)
# plt.plot(df['Temperature'], df['Cp interpolated'], label='interpolated')
# plt.plot(df['Temperature'], df['Cp calculated'], label='PyFluids')
# plt.legend()
# plt.xlabel('Temperature [K]')
# plt.ylabel('Specific Heat')
# plt.title(f'Specific Heat vs Temperature @ P={round(pressure/1e6,2)} MPa')
# plt.show()

# quit()

class geometry():
    def __init__(self, H, W, L, dia, Nch, tch):
        self.height = H
        self.width  = W
        self.length = L
        self.channel_dia = dia
        self.channel_num = Nch
        self.wall_thickness = tch

class working_fluid():
    def __init__(self, inlet, mass_flowrate):
        self.inlet = inlet
        self.flowrate = mass_flowrate
    def outlet_state(self, heat_transfer):
        deltaH = heat_transfer/self.flowrate
        self.leave = state_finder(self.inlet['pres.'], self.inlet['enth.']+deltaH, 'enth.')

class particulates():
    def __init__(self, T_in):
        self.inlet_temp = T_in
        self.density = 1442
        self.specific_heat = 0.830
        self.convective_HT = 80        

class pHX_model():
    def __init__(self, sCO2, particles):
        
        columns = [
            'q_i',
            'HTFl pres.',
            'HTFl temp.',
            'HTFl enth.',
            'HTFl m_dot',
            'HTFl Cp',

            'sand temp.',
            'sand m_dot',
            'sand Cp',
        ]





