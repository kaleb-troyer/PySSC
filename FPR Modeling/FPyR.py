
from dataclasses import dataclass, astuple, asdict, fields
import matplotlib.pyplot as plt
import scipy.optimize as opt
from ctypes import *
import pandas as pd
import numpy as np
import csv 
import sys
import os 

class designSolution(Structure):
    _fields_ = [
        ("eta", c_double),
        ("m_dot_tot", c_double),
        ("T_particle_hot_rec", c_double),
        ("Q_inc", c_double),
        ("Q_refl", c_double),
        ("Q_rad", c_double),
        ("Q_adv", c_double),
        ("Q_cond", c_double),
        ("Q_transport", c_double),
        ("Q_thermal", c_double),
        ("tauc_avg", c_double),
        ("rhoc_avg", c_double),
        ("qnetc_sol_avg", c_double),
        ("qnetw_sol_avg", c_double),
    ]

    def __repr__(self):
        
        printme = ''
        for valname, valtype in self._fields_:
            val = getattr(self, valname)
            printme += f'{valname:.<22}{float(val):.>20.3f} {self._units(valname)}\n'

        return printme

    def to_dict(self): 

        asdict = {}
        for valname, valtype in self._fields_:
            val = getattr(self, valname)
            asdict[valname] = val 

        return asdict

    def _units(self, key): 
        units = {
            "eta": "[-]",
            "m_dot_tot": "[kg/s]",
            "T_particle_hot_rec": "[K]",
            "Q_inc": "[MW]",
            "Q_refl": "[MW]",
            "Q_rad": "[MW]",
            "Q_adv": "[MW]",
            "Q_cond": "[MW]",
            "Q_transport": "[MW]",
            "Q_thermal": "[MW]",
            "tauc_avg": "[-]",
            "rhoc_avg": "[-]",
            "qnetc_sol_avg": "[MW/m²]",
            "qnetw_sol_avg": "[MW/m²]",
        }

        return units.get(key, "[?]")

@dataclass
class designParams(): 
    T_des_i: float=500.     # [C]  Receiver Inlet Temperature
    T_des_o: float=1000     # [C]  Receiver Outlet Temperature
    H_rec:   float=5.00     # [m]  Aperture Height
    W_rec:   float=10.0     # [m]  Aperture Width
    H_ratio: float=0.50     # [-]  Aperture Height Ratio
    W_ratio: float=0.50     # [-]  Aperture Width Ratio
    D_ratio: float=0.50     # [-]  Aperture Depth Ratio
    q_des_o: float=200.     # [MW] Design Receiver Output    

class FPyR(): 

    def __init__(self, path: str=os.path.join(os.getcwd(), 'FPR Modeling', 'results'), file: str='solutions.csv', newrun: bool=True): 
        
        self._path = "C:/source/.repositories/sam_dev/build/ssc/ssc/Debug"
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            self.pdll = CDLL(os.path.join(self._path, "sscd.dll"))
        elif sys.platform == 'darwin':
            self.pdll = CDLL(os.path.join(self._path, "sscd.dylib")) 
        elif sys.platform.startswith('linux'):
            self.pdll = CDLL(os.path.join(self._path, "sscd.so")) 
        else: print( 'Platform not supported ', sys.platform)

        self._pdll_types()
        self._defaults() 

        self.directory = path
        self.filename  = file 
        if newrun: # the solutions.csv file is erased, if true. 
            with open(os.path.join(self.directory, self.filename), mode='w', newline='') as file: 
                pass # create a fresh solutions file for the new class
        else: pass # otherwise, do nothing

    def simulate(self, cores: int=1, despar: designParams=designParams(), guesses: list=[[10, 10, 0.5]]): 

        def iterable(value):
            # ensures a non-iterable value is treated as one

            if not isinstance(value, (list, tuple, np.ndarray)): 
                return [value]
            else: return value

        def combinations(despar, current_obj={}, depth=0, results=None):
            # Generates a set of all combinations for the parametric study.
            
            if isinstance(despar, designParams): 
                despar = asdict(despar)
            if results is None:
                results = []

            keys = list(despar.keys())
            if depth == len(despar):
                results.append(designParams(*current_obj.values()))
                return results
            current_key = keys[depth]
            for value in iterable(despar[current_key]):
                next_obj = current_obj.copy()
                next_obj[current_key] = value
                combinations(despar, next_obj, depth + 1, results)
            return results

        def printProgressBar (iteration, total, prefix = 'Progress:', suffix = '', decimals = 1, length = 20, fill = '█', printEnd = "\r"):
            """
            Call in a loop to create terminal progress bar
            @params:
                iteration   - Required : (Int) current iteration 
                total       - Required : (Int) total iterations 
                prefix      - Optional : (Str) prefix string 
                suffix      - Optional : (Str) suffix string 
                decimals    - Optional : (Int) decimals in percent complete 
                length      - Optional : (Int) character length of bar 
                fill        - Optional : (Str) bar fill character 
                printEnd    - Optional : (Str) end character (e.g. "\r", "\r\n") 
            """
            percent = ("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
            # Print New Line on Complete
            if iteration == total: 
                print()

        allstudies = combinations(despar)
        for i, study in enumerate(allstudies): 
            for j, guess in enumerate(guesses): 
                printProgressBar(i * len(guesses) + j, len(allstudies) * len(guesses) - 1)

                results = self._optimize(study, guess)
                self._savesolution(study, results)

    def _optimize(self, study, guess): 

        def objective(x, study=study): 
            study.H_rec, study.W_rec, study.D_ratio = x

            # calculating ratios with assumed extent angles a and b
            d = study.D_ratio * study.H_rec
            a = np.deg2rad(35)
            b = np.deg2rad(80)
            h = d * np.tan(a)
            w = d * np.tan(b)
            study.H_ratio = (h + study.H_rec / 2) / study.H_rec
            study.W_ratio = 2 * w / study.W_rec 

            result = self.pdll.ssc_testing(*astuple(study))
            return 1 / result.eta 

        H_rec, W_rec, D_ratio = guess
        variables = [H_rec, W_rec, D_ratio]
        bounds = [
            (5, 30),    # [m] Receiver Height
            (5, 30),    # [m] Receiver Width
            (0.1, 1.5), # [-] Depth Ratio
        ]

        results = opt.minimize(
            objective,          # Objective function
            variables,          # Decision variables
            method='COBYLA',    # Algorithm
            bounds=bounds,      # DV Upper / Lower Bounds
            options={ 
                'tol': 0.1, 
                'maxiter': 100, 
                'rhobeg': 0.1
            }
        )

        study.H_rec = results.x[0]
        study.W_rec = results.x[1]
        optimal_instance = self.pdll.ssc_testing(*astuple(study))
        return optimal_instance

    def display(self, x: str='T_des_i', y: str='eta'): 
        df = pd.read_csv(os.path.join(self.directory, self.filename))

        plt.scatter(df[x], df[y])
        plt.show()

    def _defaults(self) -> bool: 

        self.results = None
        return True

    def _savesolution(self, study, results) -> bool: 
        '''
        Saves the current solution to the defined solutions file if
        headers are present. Otherwise, it writes the header row and 
        then saves the solution
        '''
        # checking if headers are already defined in the file
        with open(os.path.join(self.directory, self.filename), mode='r', newline='') as file: 
            reader = csv.reader(file)
            header = next(reader, None) is not None
        # writing the solution and headers, if necessary
        with open(os.path.join(self.directory, self.filename), mode='a', newline='') as file: 
            result = asdict(study) | results.to_dict()
            fields = list(result.keys())
            writer = csv.DictWriter(file, fieldnames=fields)
            if not header: 
                writer.writeheader()
            writer.writerow(result)
        
        return True

    def _pdll_types(self) -> bool: 

        self.pdll.ssc_testing.argtypes = [
            c_double,   # [C]  Receiver Inlet Temperature
            c_double,   # [C]  Receiver Outlet Temperature
            c_double,   # [m]  Aperture Height
            c_double,   # [m]  Aperture Width
            c_double,   # [-]  Aperture Height Ratio
            c_double,   # [-]  Aperture Width Ratio
            c_double,   # [-]  Aperture Depth Ratio
            c_double    # [MW] Design Receiver Output
        ]

        self.pdll.ssc_testing.restype = designSolution
        return True

if __name__=='__main__': 

    model = FPyR(newrun=False, file='solutions.csv') 
    # study = designParams(
    #     T_des_i = np.arange(500, 801, 20),
    #     T_des_o = 900,
    #     H_rec   = 15,
    #     W_rec   = 15,
    #     H_ratio = 0.2,
    #     W_ratio = 0.5,
    #     D_ratio = 0.1,
    #     q_des_o = 200,
    # )

    # guesses = [
    #     (H_rec, W_rec, 0.2) for i, (H_rec, W_rec) in enumerate(
    #         [
    #             (H, W) 
    #             for H in np.arange(5, 30, 6) 
    #             for W in np.arange(5, 30, 6) 
    #         ]
    #     )
    # ]

    # model.simulate(cores=1, despar=study, guesses=guesses)
    model.display(x='W_rec', y='eta')


