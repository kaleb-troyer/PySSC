"""
Kaleb Troyer
2024-09-03

Data generated using SolarPILOT and scipy.optimize. 

Generates figures and best-fits for solar tower, 
solar field, and receiver reduced-order models
according to power delivered to the receiver. 
"""
import matplotlib.pyplot as plt
import addcopyfighandler
import utilities as ut
import pandas as pd
import numpy as np
import os

from scipy.optimize import curve_fit

version = '1.0'
figID   = 'f0002'

def polynomial(x, a0, a1): 
    return a0 + a1 * x
def dataAnalysis(df, name, i, axes): 

    def parameterAnalysis(x, y, X, Y, j, parameter, units='m'): 
        popt, pcov = curve_fit(polynomial, x, y)
        yfit = polynomial(x, *popt)
        RMSE = np.sqrt(np.mean((y - yfit)**2))
        print(f"{parameter}")
        print(f"-> polynomial\t{popt[0]:>10.3e} + {popt[1]:5.3e}*x")
        print(f"-> RMSE      \t {RMSE:.4f}")

        # plotting
        warm = '#FC6262'
        cold = '#49AF74'
        grey = '#36454F'

        ax = axes[j, i]
        ax.scatter(x, y, label='fit data', color=cold, zorder=2, s=20) 
        ax.scatter(X, Y, label='removed', color=warm, zorder=2, s=5) 
        ax.plot(x, yfit, linestyle='--', color=grey, zorder=2)

        ax.margins(x=0)
        ax.grid(zorder=1)

        if j==3: 
            ax.set_xlabel('Power to Receiver [MW]')
        else: ax.tick_params(bottom=False)
        if i==0: 
            ax.set_ylabel(f'{parameter} [{units}]')
        else: ax.tick_params(left=False)
        if j==0: ax.set_title(f'{name}')
        if j==3 and i==3: ax.legend()

    def filtering(df): 
        condition1 = (df['q_des'] <= 200) & (df['tht'] >= 230)        
        condition2 = (df['q_des'] <= 400) & (df['tht'] >= 260)
        
        removed = df[condition1 | condition2]
        filtered = df[~(condition1 | condition2)]
        return filtered, removed

    df, removed = filtering(df)
    Preq = df['q_des']
    Htow = df['tht']
    Hrec = df['rec_height']
    Wrec = df['rec_width']
    Area = df['Simulated heliostat area']

    Preq_rm = removed['q_des']
    Htow_rm = removed['tht']
    Hrec_rm = removed['rec_height']
    Wrec_rm = removed['rec_width']
    Area_rm = removed['Simulated heliostat area']

    print(f"\n{name} analysis")
    print(f"----------------------------------------")
    # Tower Height
    parameterAnalysis(Preq, Htow, Preq_rm, Htow_rm, 0, 'Solar Tower Height')
    # Receiver Height
    parameterAnalysis(Preq, Hrec, Preq_rm, Hrec_rm, 1, 'Receiver Height')
    # Receiver Width
    parameterAnalysis(Preq, Wrec, Preq_rm, Wrec_rm, 2, 'Receiver Width')
    # Solar Field
    parameterAnalysis(Preq, Area, Preq_rm, Area_rm, 3, 'Heliostat Total Area', r'm$^{2}$')

path = os.path.join(os.getcwd(), "Figures and Data")
data = {
    "COBYLA (default)"        : "2024-08-29_reduced-order-models_COBYLA.csv",
    "COBYLA (with rhobeg=0.1)": "2024-08-29_reduced-order-models_COBYLA_initial-step.csv",
    "COBYLA (with fixed Wrec)": "2024-08-30_reduced-order-models_COBYLA_fixed.csv",
    "SLSQP (with 3-pt joc)"   : "2024-08-29_reduced-order-models_SLSQP_3-point.csv",
}

if __name__=='__main__': 

    plotter = True
    savefig = True
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(14, 14), sharex='col', sharey='row')
    for i, (key, value) in enumerate(data.items()): 

        df = pd.read_csv(os.path.join(path, "data", value))
        dataAnalysis(df, key, i, axes)

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(path, "figures", f"{figID}_V{version}_ro-models.png"), dpi=300, bbox_inches='tight')
    if plotter: plt.show()

    print("\nAnalysis complete.")
    print("")
