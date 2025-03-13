"""
Kaleb Troyer
2024-11-25

Data generated using SolarPILOT and scipy.optimize. 

Generates figures and best-fits for solar tower, 
solar field, and receiver reduced-order models
according to power delivered to the receiver. 

RO models are improved in the sense that the height
of the receiver is fixed and the optimizer is cont-
rolling aspect ratio. Receiver losses are removed
from consideration. 
"""
import matplotlib.pyplot as plt
import addcopyfighandler
import utilities as ut
import pandas as pd
import numpy as np
import os

from scipy.optimize import curve_fit

version = '3.0'
figID   = 'f0002'
savefig = False
display = True

def polynomial(x, a0, a1): 
    return a0 + a1 * x
def dataAnalysis(df, name, i, axes, show_removed=False): 

    def parameterAnalysis(x, y, X, Y, j, parameter, units='m', show_removed=show_removed): 
        popt, pcov = curve_fit(polynomial, x, y)
        yfit = polynomial(x, *popt)
        RMSE = np.sqrt(np.mean((y - yfit)**2))
        RR   = 1 - (np.sum((y - yfit)**2) / np.sum((y - np.mean(y))**2))
        print(f"{parameter}")
        print(f"-> polynomial\t{popt[0]:>10.4e} + {popt[1]:5.4e}*x")
        print(f"-> RMSE      \t {RMSE:.4f}")
        print(f"-> R**2      \t {RR:.4f}")

        # plotting
        warm = '#FC6262'
        cold = '#49AF74'
        grey = '#36454F'

        ax = axes[j, i]
        ax.scatter(x, y, label='fit data', color=cold, zorder=2, s=20) 
        if show_removed: ax.scatter(X, Y, label='removed', color=warm, zorder=2, s=5) 
        if parameter != 'Receiver Height': ax.plot(x, yfit, linestyle='--', color=grey, zorder=2)

        ax.margins(x=0)
        ax.grid(zorder=1)

        if j==2: 
            ax.set_xlabel('Power to Receiver [MW]')
        else: ax.tick_params(bottom=False)
        if i==0: 
            ax.set_ylabel(f'{parameter} [{units}]')
        else: ax.tick_params(left=False)
        if j==0: ax.set_title(f'{name}')
        if j==2 and i==0: ax.legend()

    def filtering(df): 
        bestConfig = df.loc[df.groupby('q_des')['fun'].idxmin()]

        filtered = df[df.apply(lambda row: row['fun'] <= 1.05 * bestConfig.loc[bestConfig['q_des']==row['q_des'], 'fun'].values[0], axis=1) & (df['q_des'] <= 600)]
        removed  = df[df.apply(lambda row: row['fun'] >  1.05 * bestConfig.loc[bestConfig['q_des']==row['q_des'], 'fun'].values[0], axis=1) & (df['q_des'] <= 600)]
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
    # Receiver Width
    parameterAnalysis(Preq, Wrec, Preq_rm, Wrec_rm, 1, 'Receiver Width')
    # Solar Field
    parameterAnalysis(Preq, Area, Preq_rm, Area_rm, 2, 'Heliostat Total Area', r'm$^{2}$')

path = os.path.join(os.getcwd(), "Figures and Data")
data = {
    "COBYLA (UPDATED)": "2024-11-18_reduced-order-models_UPDATED.csv",
    "COBYLA (no receiver loss)": "2024-09-04_reduced-order-models_COBYLA_no-losses.csv",
    "COBYLA (aspect ratio opt)": "2024-09-04_reduced-order-models_COBYLA_aratio.csv",
}

if __name__=='__main__': 

    ncols = len(data)

    removed = True
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(14, 14), sharex='col', sharey='row')
    for i, (key, value) in enumerate(data.items()): 

        df = pd.read_csv(os.path.join(path, "data", value))
        dataAnalysis(df, key, i, axes, show_removed=removed)

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(path, "figures", f"{figID}_V{version}_ro-models.png"), dpi=300, bbox_inches='tight')
    if display: plt.show()

    print("\nAnalysis complete.")
    print("")

