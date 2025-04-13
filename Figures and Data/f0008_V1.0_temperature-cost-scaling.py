
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.optimize import curve_fit
from pyfluids import Fluid, FluidsList, Input

def relthickness(sv, dP=24.9): 
    E1 = -np.sqrt(3) * dP + np.sqrt(3 * dP**2 + 4 * sv**2)
    E2 = 2 * sv
    return 1 - (E1 / E2)
def inner_radius(Ti=700, qd=148.2e6): 
    '''
    This function calculates the requisite inner radius
    for an assumed flow velocity and TIT. It calculates the
    mass flow rate required for the given temperature in 
    order to do so. 

    Ti: [C] Turbine inlet temperature
    qd: [W] Turbine power
    '''

    #---Assumptions
    vf = 45.72      # [m/s] sCO2 fluid velocity
    Pl = 10.0e6     # [Pa]  low pressure
    Ph = 25.0e6     # [Pa]  high pressure
    nt = 0.90       # [-]   turbine efficiency

    #---Fluid property routines
    co2_warm = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(Ti), Input.pressure(Ph)
    )

    co2_cold = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.entropy(co2_warm.entropy), Input.pressure(Pl)
    )

    h1a = co2_warm.enthalpy
    h2s = co2_cold.enthalpy
    h2a = h1a - nt * (h1a - h2s)
    co2_cold.update(Input.enthalpy(h2a), Input.pressure(Pl))

    #---Calculating mass flow rate and inner radius
    md = qd / (h1a - h2a)
    Af = md / (co2_warm.density * vf)
    ri = np.sqrt(Af / np.pi)

    return ri

class AlloySelector(): 

    def __init__(self):
        self.alloydata = {}
        self.alloys = ['316H', '800H', '740H', 'A230', 'A625']
    def select(self, alloy): 
        if alloy not in self.alloydata: 
            match alloy: 
                case '316H':
                    self.alloydata['316H'] = {
                        'stress': self.f316H, 
                        'cost': 5, # [$/kg]
                        'density': 8.00 # [kg/m3]
                    }
                case '800H': 
                    self.alloydata['800H'] = {
                        'stress': self.f800H, 
                        'cost': 23, 
                        'density': 7.94
                    }
                case '740H': 
                    self.alloydata['740H'] = {
                        'stress': self.f740H, 
                        'cost': 95, 
                        'density': 8.05
                    }
                case 'A230': 
                    self.alloydata['A230'] = {
                        'stress': self.fA230, 
                        'cost': 88, 
                        'density': 8.97
                    }
                case 'A625': 
                    self.alloydata['A625'] = {
                        'stress': self.fA625, 
                        'cost': 70, 
                        'density': 8.44
                    }

        return self.alloydata[alloy]

    def f316H(self, T, t=0, S=0, formula='s2r'): 
        if formula=='t2r': 
            cc = 17.16
            a0 = 35684.60
            a1 = -16642.65
            a2 = 7289.42
            a3 = -1475.24

            E1 = a3 * (np.log10(S))**3 + a2 * (np.log10(S))**2 + a1 * (np.log10(S)) + a0
            E2 = T + 273.15
            return 10**((E1/E2) - cc)

        elif formula=='s2r': 
            T  = T + 273.15 # [K]
            b0 = -35.27
            b1 = 47957
            b2 = 9.9400
            b3 = -15175
            return t**(T/(T*b2 + b3))/10**((T*b0 + b1)/(T*b2 + b3))
    def f800H(self, T, t=0, S=0, formula='s2r'): 
        # cc = 15.48
        # a0 = 2897.258
        # a1 = -4847.72
        # a2 = -268.31
        # a3 = 0.0

        T  = T + 273.15
        b0 = -19.870
        b1 = 36566.0
        b2 = -0.9252
        b3 = -6197.0
        return t**(T/(T*b2 + b3))/10**((T*b0 + b1)/(T*b2 + b3))
    def f740H(self, T, t=0, S=0, formula='s2r'): 
        # cc = 18.29
        # a0 = 36280.37
        # a1 = -5884.39
        # a2 = 0.0
        # a3 = 0.0

        T  = T + 273.15
        b0 = -67.74
        b1 = 87260
        b2 = 20.12
        b3 = -26560
        return t**(T/(T*b2 + b3))/10**((T*b0 + b1)/(T*b2 + b3))
    def fA230(self, T, t=0, S=0, formula='s2r'): 
        # cc = 11.28
        # a0 = 23255.67
        # a1 = -4208.21
        # a2 = 0.0
        # a3 = 0.0

        T  = T + 273.15
        b0 = -26.27
        b1 = 44158
        b2 = 4.72
        b3 = -11337
        return t**(T/(T*b2 + b3))/10**((T*b0 + b1)/(T*b2 + b3))
    def fA625(self, T, t=0, S=0, formula='s2r'): 
        # cc = 11.28
        # a0 = 23255.67
        # a1 = -4208.21
        # a2 = 0.0
        # a3 = 0.0

        T  = T + 273.15
        b0 = -44.2641
        b1 = 65825
        b2 = 12.2
        b3 = -20289
        return t**(T/(T*b2 + b3))/10**((T*b0 + b1)/(T*b2 + b3))

if __name__ == '__main__': 

    # show plots? 
    A = True
    B = True
    C = True
    D = True
    E = True
    F = True

    database = AlloySelector()
    alloys = database.alloys
    temps  = np.arange(500, 851, 50)     # [C]
    colors = sns.color_palette('mako', len(alloys))
    rivect = np.vectorize(inner_radius)
    
    #------------------------------------------#
    #---316H SS Performance According to LMP---#
    #------------------------------------------#
    for i, temp in enumerate(temps):
        stress = np.logspace(-2, 3, num=50, base=10)
        alloyinfo = database.select('316H')
        times = alloyinfo['stress'](T=temp, S=stress, formula='t2r') # [hours]
        plt.plot(times, stress, label=f'T={temp:.1f}C')

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'Stress [MPa]')
    plt.xlabel('Time to creep rupture [hr]')
    plt.title('316H LMP Performance')
    plt.legend()
    plt.grid()
    plt.margins(y=0)
    plt.tight_layout()
    if A: plt.show()
    else: plt.clf()

    #------------------------------------#
    #---Relative Thickness Coefficient---#
    #------------------------------------#
    temps  = np.linspace(400, 1000, 100) # [C]
    for i, alloy in enumerate(alloys):
        alloyinfo = database.select(alloy)
        stress = alloyinfo['stress'](T=temps, t=30*365*24*0.71) # [hours]
        factor = relthickness(stress) # [-] [th / ro]
        plt.plot(temps, factor, label=alloy, color=colors[i])

    plt.ylabel(r'Relative Thickness Coefficient, th/ro [-]')
    plt.xlabel('Temperature [C]')
    plt.title('30-year lifetime, P=25MPa')
    plt.xlim(600, 1000)
    plt.legend()
    plt.grid()
    plt.margins(x=0)
    plt.tight_layout()
    if B: plt.show()
    else: plt.clf()

    #-----------------------------------#
    #---USD Cost per length of Piping---#
    #-----------------------------------#
    for i, alloy in enumerate(alloys):
        alloyinfo = database.select(alloy)
        stress = alloyinfo['stress'](T=temps, t=30*365*24*0.71) # [hours]
        factor = relthickness(stress) # [-] [th / ro]
        
        ri = rivect(temps)
        ro = ri / (1 - factor)
        Ac = np.pi * (ro**2 - ri**2)      

        costbasis = Ac * alloyinfo['cost'] * alloyinfo['density']
        if alloy=='316H': 
            baseline = np.interp(700, temps, costbasis)
        plt.plot(temps, costbasis, label=alloy, color=colors[i])

    plt.ylabel('Cost of Piping [$/m]')
    plt.xlabel('Temperature [C]')
    plt.title('30-year lifetime, P=25MPa')
    plt.ylim(0, 300)
    plt.xlim(600, 1000)
    plt.legend()
    plt.grid()
    plt.margins(x=0)
    plt.tight_layout()
    if C: plt.show()
    else: plt.clf()

    #-------------------------------#
    #---Normalized Cost of Piping---#
    #-------------------------------#
    for i, alloy in enumerate(alloys):
        alloyinfo = database.select(alloy)
        stress = alloyinfo['stress'](T=temps, t=30*365*24*0.71) # [hours]
        factor = relthickness(stress) # [-] [th / ro]
        
        ri = rivect(temps)
        ro = ri / (1 - factor)
        Ac = np.pi * (ro**2 - ri**2)
        
        costbasis = Ac * alloyinfo['cost'] * alloyinfo['density']
        solutions = costbasis / baseline
        alloyinfo['solution'] = solutions
        plt.plot(temps, solutions, label=alloy, color=colors[i])

    plt.ylabel('Normalized Cost of Piping [-]')
    plt.xlabel('Temperature [C]')
    plt.title('30-year lifetime, P=25MPa')
    plt.ylim(0, 50)
    plt.xlim(600, 900)
    plt.legend()
    plt.grid()
    plt.margins(x=0)
    plt.tight_layout()
    if D: plt.show()
    else: plt.clf()

    #----------------------------#
    #---Minimum Cost of Piping---#
    #----------------------------#
    collector = []
    for i, alloy in enumerate(alloys):
        alloyinfo = database.select(alloy)
        collector.append(alloyinfo['solution'])
    def function(x, a0, a1, a2, a3, a4):
        tt = 30 * 365 * 24 * 0.71 

        E1 = tt**((x) / (a0*x + a1))
        E2 = 10**((a2*x + a3) / (a0*x + a1))
        return (E1 / E2) + a4

    tmin  = 660
    tmax  = 900
    array = temps[(temps >= tmin) & (temps <= tmax)]
    start = np.where(temps == array[ 0])[0][0]
    close = np.where(temps == array[-1])[0][0]
    minimums = np.minimum.reduce(collector)

    popt, pcov = curve_fit(function, temps[start:close], minimums[start:close], maxfev=5000)
    yfit = function(temps, *popt)
    RMSE = np.sqrt(np.mean((minimums[start:close] - yfit[start:close])**2))

    var  = 0.07
    base = 0.05
    def f(p): 
        return np.abs(p - base) / var
    def p(f): 
        v = var
        b = base
        return (f*v + b) 

    b620 = (620, 0.044)
    b760 = (760, 0.120)
    r620 = (620, 0.077)
    r760 = (760, 0.239)
    pnts = np.array([b620, b760, r620, r760])
    plt.plot(temps, minimums, label='Normalized Pipe Costs', color=colors[0])
    plt.scatter(pnts[2:4, 0], f(pnts[2:4, 1]), marker='x', label='Recuperated Cycle (White, 2017)')

    plt.ylabel('Normalized Cost of Piping [-]')
    plt.xlabel('Temperature [C]')
    plt.title('30-year lifetime, P=25MPa')
    plt.ylim(0, 5)
    plt.xlim(550, 800)
    plt.legend()
    plt.grid()
    plt.margins(x=0)
    plt.tight_layout()
    if E: plt.show()
    else: plt.clf()

    #------------------------#
    #---Piping Cost Factor---#
    #------------------------#
    plt.plot(temps, p(minimums), color=colors[0], label='Cost Factor')
    plt.scatter(pnts[2:4, 0], pnts[2:4, 1], marker='x', label='Recuperated Cycle (White, 2017)')

    plt.ylabel('Piping Costs / Power Block Costs [-]')
    plt.xlabel('Temperature [C]')
    plt.title(f'$p={var}f + {base}$')
    plt.legend()
    plt.ylim(0, 0.8)
    plt.xlim(550, 850)
    plt.grid()
    plt.margins(x=0)
    plt.tight_layout()
    if F: plt.show()
    else: plt.clf()



