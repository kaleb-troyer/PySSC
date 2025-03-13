
from pyfluids import Fluid, FluidsList, Input 
from Pyflow.pyflow import Model, Pyflow
from PyHXsim import Sand
import numpy as np

def count_channels_cold(W, D=0.006, d=0.006): 
    return W / (D + d)
def relative_roughness(D=0.006, mat='SiC', geo='TO'): 
    if mat == 'SiC' and geo == 'TO': 
        ew = 5e-4 # [m] wall roughness, meters
        rr = ew / D 
    elif mat == 'SiC' and geo == 'FP': 
        ew = 5e-6   # [m]
        rr = ew / D 
    elif mat == '316H' and geo == 'TO': 
        ew = 10e-6  # [m]
        rr = ew / D 
    elif mat == '316H' and geo == 'FP': 
        ew = 1e-6
        rr = ew / D
    else: raise ValueError(f"Material or geometry not recognized.")
    return rr
def conductivity(T, mat='SiC'): 
    if mat == 'SiC': 
        # Data taken from NIST ceramics data portal. 
        # https://srdata.nist.gov/CeramicDataPortal/Pds/Scdscs
        temp = np.array([20, 500, 1000, 1200, 1400, 1500])    # [C]
        cond = np.array([114, 55.1, 35.7, 31.3, 27.8, 26.3])  # [W/m-K]
    elif mat == '316H': 
        # https://www.govinfo.gov/content/pkg/GOVPUB-C13-bb31c3f26e1bbf969c51d7614aeba8d1/pdf/GOVPUB-C13-bb31c3f26e1bbf969c51d7614aeba8d1.pdf
        temp = np.array(
            [
                90, 130, 170, 210, 250, 290, 330, 370, 410, 450, 
                490, 530, 570, 610, 650, 690, 730, 770, 810, 850
            ]
        )
        cond = np.array(
            [
                14.9, 15.5, 16.2, 16.8, 17.4, 18.1, 18.7, 
                19.3, 19.9, 20.5, 21.1, 21.8, 22.4, 23.0, 
                23.6, 24.3, 24.9, 25.6, 26.2, 26.9
            ]
        )
    else: raise ValueError(f"Material ({mat}) not recognized. Use 'SiC' or '316H'.")
    return np.interp(T, temp, cond)

def routine(mat='SiC', geo='FP'): 
    T_warm_avg = 1010
    T_cold_avg = T_warm_avg - 280

    A_warm = 0.0018
    wp = 0.03
    hp = 0.06
    th = 0.003
    Di = 0.006
    rr = relative_roughness(mat=mat, geo=geo)

    W = wp
    dxi = hp
    Nch = count_channels_cold(W)

    sand = Sand().update(temperature=T_warm_avg)
    sco2 = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(T_cold_avg), Input.pressure(25e6 - 100e3)
    )

    air = Fluid(FluidsList.Air).with_state(
        Input.temperature(T_warm_avg), Input.pressure(101325)
    )

    pipe = Pyflow(
        Model.Pipe(D=Di, L=hp, massflow=0.0010, e=rr), 
        fluid = sco2
    )

    htc = 340
    khx = conductivity(T=(T_warm_avg + T_cold_avg)/2, mat=mat)

    R_cold_i = 1 / (Nch * np.pi * Di * pipe.htc * dxi)
    R_warm_i = 1 / (2 * W * htc * dxi)
    R_cond_i = ((th) / 2) / (2 * W * khx * dxi)

    UA = (R_warm_i + R_cond_i + R_cold_i)**(-1)
    return UA

# UAfpSiC = routine(mat='SiC', geo='FP')
# UAtoSiC = routine(mat='SiC', geo='TO')
# UAfp316 = routine(mat='316H', geo='TO')

# print(f"UAfpSiC = {UAfpSiC:.2f} W/K")
# print(f"UAtoSiC = {UAtoSiC:.2f} W/K")
# print(f"UAfp316 = {UAfp316:.2f} W/K")

pipe = Pyflow(
    Model.Pipe(D=0.003, L=0.1, massflow=0.01, e=(5e-4)/0.003), 
    fluid := Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(700), Input.pressure(25e6)
    )
)

print(pipe)


