import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import table_builder, state_finder, get_precise_table
from scipy.optimize import minimize

## user inputs ##
path  = 'co2_tables'

# Heat Exchanger Dimensions
W = 200     # [m] total width of heat exchanger
L = 200     # [m] total length of heat exchanger
H = 200     # [m] total height of heat exchanger
Nch = 100   # channel count for each side
tch = 0.002 # [m] wall thickness
k = 70      # [W/m*K] coeff. of thermal conductivity

# sCO2 properties and inputs
Tf_lo = 830.51  # [K] after recuperation
Tf_hi = 1023.15 # [K] TIT = 750C
Pf_hi = 25.00   # [MPa]
P_del = 0.005   # pressure delta / pressure
mf_f  = 758.27  # [kg/s] mass flow rate of HTF
h_bar_f = 300   # [W/K*m^2] sCO2 HT coeff.

# particle properties and inputs
Tp_lo = None    # [K] after heat exchange
Tp_hi = 1300.0  # [K] HX inlet temperature
rho_p = 1442.0  # [kg/m^3] bulk density loose sand
cp_p  = 0.830   # [kJ/kg*K] silica sand 
h_bar_p = 80.0  # [W/K*m^2] course silica sand

# e-NTU method inputs
N = 10
Rtot = ((1 / h_bar_f) + (tch / (k * W)) + (1 / h_bar_p))

columns = [
    'q_i',

    'HTF pres.', 
    'HTF temp.',
    'HTF enth.',
    'HTF mass flow', 
    'HTF Cp',

    'Particle temp.', 
    'Particle mass flow', 
    'Particle Cp'
]

v = 0.00325 # [m/s]
A = (W / (Nch * 2)) - tch
mf_p = v * rho_p * A * Nch
print(f"Area: {A}")
print(f"mf_p: {mf_p:7.3f}")

node_data = pd.DataFrame(0, index=range(1, N+1), columns=columns)
pressure_drop = P_del * Pf_hi
pressure_table_hi = get_precise_table(table_builder(Pf_hi), Tf_hi, Tf_lo)
pressure_table_lo = get_precise_table(table_builder(Pf_hi - pressure_drop), Tf_hi, Tf_lo)
inlet_state = state_finder(Pf_hi, Tf_lo, 'temp.')
leave_state = state_finder(Pf_hi - pressure_drop, Tf_hi, 'temp.')

q_total = mf_f * (leave_state['enth.'] - inlet_state['enth.'])
q_i = q_total / (N - 1)

# setting particle node values
def build_HX(mf_p=mf_p, N=N, cp_p=cp_p, mf_f=mf_f, q_i=q_i, Nch=Nch, W=W, Rtot=Rtot, L=L, Tp_hi=Tp_hi):

    node_data.loc[1, 'q_i'] = q_i
    node_data.loc[1, 'HTF pres.'] = Pf_hi
    node_data.loc[1, 'HTF temp.'] = Tf_lo
    node_data.loc[1, 'HTF enth.'] = inlet_state['enth.']
    node_data.loc[1, 'HTF mass flow'] = mf_f
    node_data.loc[1, 'HTF Cp'] = inlet_state['Cp']
    node_data.loc[1, 'Particle Cp'] = cp_p
    node_data.loc[N, 'Particle temp.'] = Tp_hi

    # setting HTF node values
    for i in range(2, N+1): 
        ith_pressure = Pf_hi - ((i-1) * pressure_drop / (N-1))

        h_last = node_data.loc[i-1, 'HTF enth.']
        h_this = (q_i / mf_f) + h_last 

        node_state = state_finder(ith_pressure, h_this, 'enth.')

        node_data.loc[i, 'q_i'] = q_i
        node_data.loc[i, 'HTF pres.'] = ith_pressure
        node_data.loc[i, 'HTF temp.'] = node_state['temp.']
        node_data.loc[i, 'HTF enth.'] = node_state['enth.']
        node_data.loc[i, 'HTF mass flow'] = mf_f
        node_data.loc[i, 'HTF Cp'] = node_state['Cp']
        node_data.loc[i, 'Particle Cp'] = cp_p

    for i in range(1, N+1): node_data.loc[i, 'Particle mass flow'] = mf_p
    for i in range(1, N):
        row = N - i

        T_last = node_data.loc[row+1, 'Particle temp.']
        T_this = T_last - (q_i / (mf_p * cp_p))
        node_data.loc[row, 'Particle temp.'] = T_this


    # print(node_data.loc[:, ['Particle temp.', 'HTF temp.']]) 
    # quit()

    columns = ['inlet', 'leave', 'eff_i', 'NTU_i', 'UA_i', 'dx_i']
    subdivision_data = pd.DataFrame(0, index=range(1, N), columns=columns)
    for i in range(1, N): 

        # print(node_data)
        # quit()

        Cmin = min(mf_f * node_data.loc[i,'HTF Cp'], mf_p * cp_p)
        Cmax = max(mf_f * node_data.loc[i,'HTF Cp'], mf_p * cp_p)
        cold_side_temp = node_data.loc[i, 'HTF temp.']
        warm_side_temp = node_data.loc[i+1, 'Particle temp.']

        # print(Cmin)
        # print(warm_side_temp)
        # print(cold_side_temp)
        # quit()

        eff_i = q_i / (Cmin * (warm_side_temp - cold_side_temp))

        Cri = Cmin / Cmax
        # if eff_i - 1 < 0:
        #     return [10, subdivision_data]
        
        # print(eff_i)
        # print(Cri)
        # quit()

        NTU_i = (1 / (1 - Cri)) * np.log(((eff_i * Cri) - 1) / (eff_i - 1))
        UA_i = NTU_i * Cmin

        dx_i = (UA_i / (2 * Nch * W)) * Rtot

        subdivision_data.loc[i, 'inlet'] = i
        subdivision_data.loc[i, 'leave'] = i+1
        subdivision_data.loc[i, 'eff_i'] = eff_i
        subdivision_data.loc[i, 'NTU_i'] = NTU_i
        subdivision_data.loc[i, 'UA_i'] = UA_i
        subdivision_data.loc[i, 'dx_i'] = dx_i
    
    dx = sum(dx_i for dx_i in subdivision_data['dx_i'])
    error = ((dx - L)**2) / L

    return [error, subdivision_data]
def validate_HX(mf_p):
    result = build_HX(mf_p)
    error = result[0]

    return error

Tin_p_guess = 1500
# bounds = [(450, 700)]

# mf_p = minimize(validate_HX, mf_p_guess, bounds=bounds).x[0]

subdivision_data = build_HX(Tp_hi=Tin_p_guess)[1]
UA_all = sum(UA for UA in subdivision_data['UA_i'])

print('\n')
print(f'        Total UA = {UA_all:>7.2f} [W/K]')
print(f'  Mass Flow Rate = {mf_p:>7.2f} [kg/s]')
print(f' Volumetric Flow = {mf_p/rho_p:>7.2f} [m^3/s]')
print(f'  HT Coefficient = {h_bar_p:>7.2f} [W/K*m^2]')
print(f'    Contact Area = {UA_all / h_bar_p:>7.2f} [m^2]')
print('\n')

print(node_data)

HTF = node_data['HTF temp.']
particles = node_data['Particle temp.']
plt.plot(HTF, color = 'blue')
plt.plot(particles, color = 'red')
plt.show()


