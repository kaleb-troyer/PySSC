
from dataclasses import dataclass, astuple, asdict
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.optimize as opt
from queue import Empty
from ctypes import *
import pandas as pd
import numpy as np
import time
import csv
import sys
import os

from copy import deepcopy as dc

# for benchmarking the program speed
class timer():
    def __init__(self, quiet=False):
        self.start = 0
        self.bench = 0
        self.quiet = quiet
        self.timed = False
        self.lap = 0
    def tic(self):
        if self.timed == 0:
            self.timed = True
            self.start = time.time()
            if not self.quiet: print("\nStarting timer...")
        else: 
            self.lap = time.time()
            if not self.quiet: print(f"{self.lap - self.start:.4f}")
            return self.lap - self.start
    def toc(self):
        if self.timed:
            self.timed = False
            self.bench = time.time()
            if not self.quiet: print(f"time elapsed: {self.bench-self.start:.4f}")
        else: pass
        return self.bench-self.start

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

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        raise TypeError("Key must be a string corresponding to a field name.")

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
            "T_particle_hot_rec": "[C]",
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
    T_des_i: float=500      # [C]  Receiver Inlet Temperature
    T_des_o: float=1000     # [C]  Receiver Outlet Temperature
    H_rec:   float=1.0      # [m]  Aperture Height (optimized)
    W_rec:   float=1.0      # [m]  Aperture Width (optimized)
    H_ratio: float=1.0      # [-]  Curtain height / aperture height (assigned in optimizer) 
    W_ratio: float=1.0      # [-]  Curtain width / aperture width (assigned in optimizer) 
    D_ratio: float=0.0      # [-]  Aperture depth / aperture height (assigned in optimizer)
    q_des_o: float=200      # [MW] Design Receiver Output 

class Receiver(): 

    def __init__(self):
        _path = os.path.join(os.getcwd(), 'SSC CSP API', 'core')
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            self.pdll = CDLL(os.path.join(_path, "sscd.dll"))
        elif sys.platform.startswith('linux'):
            self.pdll = CDLL(os.path.join(_path, "ssc.so")) 
        else: print( 'Platform not supported ', sys.platform)

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

    def solve(self, study, guess): 

        study.H_rec   = guess[0]
        study.W_rec   = study.H_rec

        study.H_ratio = guess[1]
        study.W_ratio = study.H_ratio

        return self.pdll.ssc_testing(*astuple(study))

    def optimize(self, study, guess): 

        # H_rec, r_curtain, R = guess
        def objective(x, study=study): 
            '''
            To maximize efficiency by modifying curtain dimensions. 
            '''

            result = self.solve(study, x)
            # print(f'({x[0]:5.2f}, {x[1]:4.2f}) -> {100*result.eta:5.2f}%')
            return 1 / result.eta

        variables = [guess[0], guess[1]]
        bounds = [
            (0, 15),    # [m] Receiver Height
            (0.2, 1.5)  # [-] Receiver Height Ratio
        ]

        results = opt.minimize(
            objective,
            variables,
            method='COBYLA',
            bounds=bounds, 
            options={ 
                'tol': 0.01, 
                'maxiter': 200, 
                'rhobeg': 0.1
            }
        )

        study.H_rec = results.x[0]
        study.W_rec = results.x[0]
        optimal_instance = self.pdll.ssc_testing(*astuple(study))
        return optimal_instance

def multiprocesing(despar, guesses, cores): 

    def isiterable(value):
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
        for value in isiterable(despar[current_key]):
            next_obj = current_obj.copy()
            next_obj[current_key] = value
            combinations(despar, next_obj, depth + 1, results)
        return results

    studies = [study for study in combinations(despar) if study.T_des_i < study.T_des_o]
    manager = mp.Manager()
    mpqueue = manager.Queue()
    tracker = manager.Value('i', 0)
    cpupool = mp.Pool(processes=cores)
    watcher = mp.Process(target=mplistener, args=(
            mpqueue, tracker, len(studies) * len(guesses), cores
        )
    )
    
    watcher.start()
    try: # using a try-finally block to ensure resources are always released

        dispatch = [
            cpupool.apply_async(mpworker, (study, guess, mpqueue, tracker))
            for study in studies
            for guess in guesses
        ]

        for task in dispatch: 
            task.get()

        mpqueue.put('kill')
        watcher.join() 

    finally: # ensuring resources are released
        cpupool.close()
        cpupool.join() 
        manager.shutdown()

def mpworker(study, guess, queue=None, tracker=None):
    '''
    Process for executing one study of the parametric process. 
    ''' 
    
    clock = timer(quiet=True)
    clock.tic()

    receiver = Receiver()
    try: # attempting to optimize the receiver
        results = receiver.optimize(study, guess)
    except Exception as e: 
        results = str(e)
    elapsed = clock.toc()

    tracker.value += 1
    queue.put((study, results, elapsed))

def mplistener(queue, tracker, total, cores): 
    '''
    Process for saving and displaying the results of each multithreaded
    parametric process. 
    '''

    def writer(message): 
        
        for line in message: 
            sys.stdout.write(line)
        for _ in range(len(message)-1): 
            sys.stdout.write('\033[F')
        sys.stdout.flush()

    def getlines(): 

        display = [
            ("FPR efficiency", "eta"),
            ("HTF mass flow rate", "m_dot_tot"),
            ("outlet temperature", "T_particle_hot_rec"), 
        ]

        return display
    
    def timeleft(i, N, times, cores=cores):
        
        average = np.nanmean(times)
        remaining_iters = N - i
        remaining_time = average * remaining_iters / (cores - 1)

        if remaining_time > 84600: 
            message = f'{remaining_time / 84600:.2f} days'
        elif remaining_time > 3600: 
            message = f'{remaining_time / 3600:.2f} hours'
        elif remaining_time > 60: 
            message = f'{remaining_time / 60:.2f} minutes' 
        elif remaining_time > 0: 
            message = f'{int(remaining_time)} seconds'         
        else: message = 'complete'

        return message

    display = getlines()
    timeset = np.full(total, np.nan)
    message = ['Initializing parametric study...', '\n', '\n']
    
    new = True

    while True: 

        writer(message)
        try: results = queue.get(timeout=2) 
        except Empty: 
            continue
        
        firstline = f' study {tracker.value:03} / {total:03} '
        if results=='kill': 
            for _ in range(len(asdict(study))+len(display)+5): sys.stdout.write('\n')
            break
        elif isinstance(results, tuple): 

            message = [f'#-{firstline:-^41}-#']
            # unpacking the queued package 
            study    = results[0]
            solution = results[1]
            elapsed  = results[2]

            # parsing the optimal design 
            timeset[tracker.value - 1] = elapsed
            message.append(f'\n{"Time remaining":.<30}{timeleft(tracker.value, total, timeset):.>15}')
            message.append('\n')
            for key, value in asdict(study).items(): 
                message.append(f'\n{key:.<35}{value:.>10.4f}')
            message.append('\n')

            try: # attempting to save the results
                for line in display: 
                    message.append(f'\n{line[0]:.<35}{solution[line[1]]:.>10.4f}')
                savesolution(study, solution, new)
                new = False
            except KeyError: 
                for line in display:                 
                    message.append(f'\n{line[0]:.<35}{"nan":.>10}')

            message.append('\n')
            writer(message)

def savesolution(study, results, new, path: str=os.path.join(os.getcwd(), 'FPR Modeling', 'results'), file: str='solutions.csv'): 
    '''
    Saves the current solution to the defined solutions file if
    headers are present. Otherwise, it writes the header row and 
    then saves the solution
    '''

    directory = path 
    filename  = file
    if new: # the solutions.csv file is erased, if true. 
        with open(os.path.join(directory, filename), mode='w', newline='') as f: 
            pass # create a fresh solutions file for the new class
    else: pass # otherwise, do nothing

    # checking if headers are already defined in the file
    with open(os.path.join(directory, filename), mode='r', newline='') as f: 
        reader = csv.reader(f)
        header = next(reader, None) is not None
    # writing the solution and headers, if necessary
    with open(os.path.join(directory, filename), mode='a', newline='') as f: 
        result = asdict(study) | results.to_dict()
        fields = list(result.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        if not header: 
            writer.writeheader()
        writer.writerow(result)

if __name__=='__main__': 

    studies = designParams(
        T_des_i=np.arange(550, 1051, 50),
        T_des_o=np.arange(550, 1051, 50),
        q_des_o=np.arange(200,  601, 50),
        W_ratio=1.5, # optimized
        H_ratio=1.5, # optimized
        D_ratio=0.2  # static (doesn't matter)
    )

    guesses = [
        (s, r) for i, (s, r) in enumerate(
            [ (s, r) 
                for s in np.linspace(3, 15, 3) # [m*1e-2] Receiver Size guesses
                for r in np.linspace(0.2, 1.5, 3) # [m*1e-1] Receiver Aspect Ratio guesses
            ]
        )
    ]

    multiprocesing(studies, guesses, cores=6)

    #----------------------------------------------#
    #--------- single-core simplification ---------#
    #----------------------------------------------#

    # study = designParams(
    #     T_des_i=600, 
    #     T_des_o=900, 
    #     q_des_o=300, 
    #     W_ratio=1.0, 
    #     H_ratio=1.2, 
    #     D_ratio=0.2
    # )

    # guess = 15

    # new = True
    # receiver = Receiver()
    # for i in np.linspace(3.0, 12.0, 20):        # sizes
    #     for j in np.linspace(0.2, 1.8, 20):     # ratios
    #         # study.H_ratio = j
    #         solution = receiver.optimize(study, (i, j))

    #         print(f'Solution converged: ({study.H_rec:5.2f}, {study.H_ratio:4.2f}) -> {100*solution.eta:5.2f}%\n')
    #         savesolution(study, solution, new, file='temp.csv')
    #         new = False

    #----------------------------------------------#
    #----------- might want this later ------------#
    #----------------------------------------------#

    # guesses = [
    #     (H, r, R) for i, (H, r, R) in enumerate(
    #         [ (H, r, R)
    #             for H in np.arange(20, 28,  3) # [m*1e-2] Tower Height guesses
    #             for r in np.arange( 3, 13,  3) # [m*1e-1] Receiver Aspect Ratio guesses
    #             for R in np.arange(15, 16, 10)
    #         ]
    #     )
    # ]

