
from scipy.integrate import quad 
import scipy.constants as const
import pandas as pd
import numpy as np 
import warnings
import os

# Suppress integration warnings
warnings.simplefilter("ignore", category=UserWarning)

def n_sand(la): 
    n = np.sqrt(1 + 0.6961663 / (1-(0.0684043 / la)**2) + 0.4079426 / (1 - (0.1162414 / la)**2) + 0.8974794 / (1 - (9.896161 / la)**2))
    return n

def Eb_la(la, n=1.0, T=500): 
    """
    Spectral function (SBEP) as a function of wavelength (la).
    """
    N = 2 * np.pi * const.h * const.c**2
    E = np.clip(const.h * const.c / (n * la * const.k * T), None, 700)
    D = n**2 * la**5 * (np.exp(E) - 1)
    return N / D

def get_reflectance(n2, k2, n1=1.0, th=0): 
    
    th = np.deg2rad(th)
    p = np.sqrt((1/2) * (
            np.sqrt(
                (n2**2 - k2**2 - n1**2 * np.sin(th)**2)**2 + 4 * n2**2 * k2**2
            ) + (n2**2 - k2**2 - n1**2 * np.sin(th)**2)
        )
    )

    q = np.sqrt((1/2) * (
            np.sqrt(
                (n2**2 - k2**2 - n1**2 * np.sin(th)**2)**2 + 4 * n2**2 * k2**2
            ) - (n2**2 - k2**2 - n1**2 * np.sin(th)**2)
        )
    )

    rhTE = ((n1 * np.cos(th) - p)**2 + q**2) / ((n1 * np.cos(th) + p)**2 + q**2)
    rhTM = (
        ((p - n1 * np.sin(th) * np.tan(th))**2 + q**2)
         / ((p + n1 * np.sin(th) * np.tan(th))**2 + q**2)
    ) * rhTE

    return (rhTE + rhTM) / 2

def get_semitransparent(rho, la, th): 

    th = np.deg2rad(th)
    di = 0.1
    ki = 0.2
    Ki = 4 * np.pi * ki / (la * 1e-6)
    tau = np.exp(-Ki * di / np.cos(th))

    Tset = (tau * (1 - rho)**2) / (1 - rho**2 * tau**2)
    Rset = rho * (1 + (((1 - rho)**2 * tau**2) / (1 - rho**2 * tau**2)))
    Aset = 1 - Tset - Rset
    return Tset, Rset, Aset

def ep_la(la, df, th):
    '''
    la1 [um]    band wavelength start
    la2 [um]    band wavelength close
    '''

    n = np.interp(la, df['la'], df['n'])
    k = np.interp(la, df['la'], df['k'])
    rh_la = get_reflectance(n, k, th=th)
    if k==0: 
        Tset, Rset, Aset = get_semitransparent(rh_la, la, th)
        return Aset
    else: 
        return 1 - rh_la

def get_emittance(la1=0, la2=np.inf, T=500, material='Nickel', angle=0): 
    '''
    T   [K]     material temperature
    la1 [um]    band wavelength start
    la2 [um]    band wavelength close
    '''

    if material not in ['Nickel', 'Chromium', 'Tungsten', 'Iron', 'Cobalt', 'Silica', 'Alumina']: 
        raise ValueError(f'Material ({material}) not recognized.')

    file = f'{material}.csv'
    data = pd.read_csv(os.path.join(os.getcwd(), 'FPR Optics', 'data', file))
    
    la1 = np.maximum(la1, data['la'].min()) * 1e-6
    la2 = np.minimum(la2, data['la'].max()) * 1e-6  

    resultA, errorA = tuple(map(sum, zip(
        quad(lambda x: Eb_la(la=x, T=T) * ep_la(la=x * 1e6, df=data, th=angle), la1, la2 / 2), 
        quad(lambda x: Eb_la(la=x, T=T) * ep_la(la=x * 1e6, df=data, th=angle), la2 / 2, la2)
    )))

    resultB, errorB = tuple(map(sum, zip(
        quad(lambda x: Eb_la(la=x, T=T), la1, la2 / 2), 
        quad(lambda x: Eb_la(la=x, T=T), la2 / 2, la2)
    )))

    errorp = 100 * errorA / resultB
    if errorp >= 0.5: 
        print(f"Error: +/- {errorp:.2f}%")
    return resultA / resultB

if __name__=='__main__': 

    Tsun = 1000     # [K]   for absorptance, use origin temperature
    la1 = 0.
    la2 = 20

    # frNi = 0.57     # [-]   Hanes 230 alloy composition: Nickel fraction
    # frCr = 0.22     # [-]   Hanes 230 alloy composition: Chromium fraction
    # frW_ = 0.14     # [-]   Hanes 230 alloy composition: Tungsten fraction
    # frFe = 0.02     # [-]   Hanes 230 alloy composition: Iron fraction
    # frCo = 0.05     # [-]   Hanes 230 alloy composition: Cobalt fraction

    # epNi = get_emittance(la1=la1, la2=la2, T=Tsun, material='Nickel')
    # epCr = get_emittance(la1=la1, la2=la2, T=Tsun, material='Chromium')
    # epW_ = get_emittance(la1=la1, la2=la2, T=Tsun, material='Tungsten')
    # epFe = get_emittance(la1=la1, la2=la2, T=Tsun, material='Iron')
    # epCo = get_emittance(la1=la1, la2=la2, T=Tsun, material='Cobalt')
    # total = epNi * frNi + epCr * frCr + epW_ * frW_ + epFe * frFe + epCo * frCo

    # print(f'Nickel   = {epNi*100:5.2f}% (wt = {frNi*100:2.0f}%)')
    # print(f'Chromium = {epCr*100:5.2f}% (wt = {frCr*100:2.0f}%)')
    # print(f'Tungsten = {epW_*100:5.2f}% (wt = {frW_*100:2.0f}%)')
    # print(f'Iron     = {epFe*100:5.2f}% (wt = {frFe*100:2.0f}%)')
    # print(f'Cobalt   = {epCo*100:5.2f}% (wt = {frCo*100:2.0f}%)')
    # print(f'Total Absorptivity = {100*total:.2f}%')
    
    epSiO2 = get_emittance(la1=la1, la2=la2, T=Tsun, material='Alumina')
    print(f'Alumina = {epSiO2*100:5.2f}%')


