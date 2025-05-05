
from core import sco2_cycle as solver
from core import sco2_plots as cy_plt
from queue import Empty
import multiprocessing as mp
import pandas as pd
import numpy as np
import time
import sys
import csv
import os

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

class System(): 
    '''
    A PySSC wrapper for clean data structure management, file saving, and 
    multithreaded parametric studies. 

    ### Attributes

    path : str=str=os.path.join(os.getcwd(), 'SSC CSP API', 'results')
        Path to the folder of the data file.
    file : str='solutions.csv'
        Name of the csv. If no file exists, then one is created. 
    newrun : bool=True
        If true, the file is overwritten when the class is created. 
    
    ### Methods
    ```
    update(params: dict={}, prebuilt: str='maxeta') -> None 
        Dynamically updates the set of design parameters that constrain the optimization. 
    optimize(savefig: bool=False, file: str='', name: str='') -> dict
        Initializes the system-level optimization via SAM Simulation Core. 
    parametric(params: dict={}, cores: int=1) -> None
        Initializes a parametric study of parameterized system-level optimizations. 
    ```
    '''
    def __init__(self, path: str=os.path.join(os.getcwd(), 'SSC CSP API', 'results'), file: str='solutions.csv', newrun: bool=True) -> None:

        self.parameters = {}
        self.solution   = {}
        self.solutions  = []
        self.solver     = solver.C_sco2_sim(1)
        self.figurename = 'unconfigured'
        self.filename   = file
        self.directory  = path
        self.studyid  = ''

        if newrun: # the solutions.csv file is erased, if true. 
            with open(os.path.join(self.directory, self.filename), mode='w', newline='') as file: 
                pass # create a fresh solutions file for the new class
        else: # otherwise, previous solutions are loaded into the class
            df = pd.read_csv(os.path.join(self.directory, self.filename))
            self.solutions = df.to_dict('records')
            self.solution  = self.solutions[-1]
        self._defaults()
    def __str__(self) -> str:

        formattedstr = []
        for key in sorted(self.solution.keys()):
            value = self.solution[key]
            try:
                formattedstr.append(f'{key:.<36}{value:.>10.4f}')
            except (TypeError, ValueError) as e:
                formattedstr.append(f'{key:.<36}{"[...]":.>10}')
        return '\n'.join(formattedstr)

    def update(self, params: dict={}, prebuilt: str='maxeta') -> None: 
        '''
        Updating the design parameters using either a user-defined dictionary 
        or a prebuilt configuration.

        ### Parameters

        params : dict={}
            (optional) key-value pairs to update design input parameters. 
        prebuilt : str='maxeta'
            (optional) apply predefined design input parameters.  
        '''

        if not params and prebuilt=='maxeta': # if optimizing efficiency...
            self.studyid = 'Eta-maximized'
            self.figurename = 'cycle_design_efficiency'
            self.parameters['des_objective']    = 1
            self.parameters['UA_recup_tot_des'] = 15 * 1000 * (self.parameters["W_dot_net_des"]) / 50.0
        elif not params and prebuilt=='minlcoe': # if optimizing LCOE...
            self.studyid = 'CSP Gen3'
            self.figurename = 'cycle_design_lcoe'
            self.parameters['des_objective']    = 3
            self.parameters['UA_recup_tot_des'] = -100e3    # [kW/K] 
        else: # adding custom parameters
            self.parameters.update(params)
    def optimize(self, savefig: bool=False, figurename: str='', studyid: str='') -> dict: 
        '''
        Performs a single study according to the current design parameters. 
        The solution is saved in the defined solutions file.

        ### Parameters

        savefig : bool=False
            (optional) determines whether the T-S diagram is saved. 
        figurename : str=''
            (optional) specifies the file name of the T-S diagram. 
        studyid : str=''
            (optional) specifies the reference ID of the current study. 
        '''
        if figurename != '': self.figurename = figurename
        if studyid != '': self.studyid  = studyid
        self.solver.overwrite_default_design_parameters(
            self.parameters
        )

        try: # attempting to solve the cycle
            self.solver.solve_sco2_case()
            self.solution = self.solver.m_solve_dict
            self.solution['label'] = self.studyid
            self.solutions.append(self.solution.copy())

            if savefig: # generating and saving a TS diagram, if true
                self.figure = cy_plt.C_sco2_TS_PH_plot(
                    self.solution
                )

                self.figure.is_save_plot = True
                self.figure.file_name = self.figurename
                self.figure.plot_new_figure()
        except: print('Solution not found. Optimization failed to converge.\n')

        self._savesolution()
        return self.solution
    def parametric(self, params: dict={}, cores: int=1) -> None:
        '''
        Conducts a parametric study by breaking the specified parameters into 
        a complete set of combinations and performing a study one-by-one. 

        Example
        ```
        system = System(newrun=True)
        system.update(prebuilt='maxeta')

        system.parametric(params={
            'dT_PHX_hot_approach' : np.arange(20, 201, 10), 
            'T_htf_hot_des'       : np.arange(680, 901, 10)
        }, cores=4)
        ```

        ### Parameters
        
        params : dict={}
            (optional) defines the design-space of the parametric study. 
        cores : int=1
            (optional) optionally enable multithreading. 
        '''
        def combinations(d, current_combination={}, depth=0, results=None):
            # Generates a set of all combinations for the parametric study.
            
            if results is None:
                results = []
            keys = list(d.keys())
            if depth == len(d):
                results.append(current_combination)
                return results
            current_key = keys[depth]
            for value in d[current_key]:
                next_combination = current_combination.copy()
                next_combination[current_key] = value
                combinations(d, next_combination, depth + 1, results)
            return results

        self.studies = combinations(params)
        self.solutions = []

        if cores <= 1: 
            # simple procedure if not multithreading

            print("Initializing parametric study...\n")
            for i, study in enumerate(self.studies):
                print(f'#------------ study {i+1:03} / {len(self.studies):03} --------------#')
                for key, value in study.items(): 
                    print(f'{key:.<35}{value:.>10.4f}')
                print('')
                self.update(study)
                self.optimize(savefig=False)

            return True
        elif cores >= 2: 
            # preparing all jobs and the job manager for multithreading. 

            manager = mp.Manager()
            parlock = manager.Lock()
            mpqueue = manager.Queue()
            tracker = manager.Value('i', 0)
            cpupool = mp.Pool(processes=cores)
            watcher = cpupool.apply_async(self._mplistener, (mpqueue, tracker, len(self.studies), cores))

            try: # using a try-finally block to ensure resources are always released
                
                jobs = []
                for study in self.studies: 
                    job = cpupool.apply_async(self._mpworker, (study, mpqueue, tracker, parlock))
                    jobs.append(job)
                for job in jobs: 
                    job.get()
                mpqueue.put('kill')
                watcher.get()

            finally:

                cpupool.close()
                cpupool.join()
                manager.shutdown()

            return True

    def _mpworker(self, study, queue, tracker, parlock):
        '''
        Process for executing one study of the parametric process. 
        '''

        clock = timer(quiet=True)

        clock.tic()
        opt_solver = solver.C_sco2_sim(1)
        with parlock: 
            self.update(study)
            opt_solver.overwrite_default_design_parameters(
                self.parameters
            )

        try: # attempting to solve the cycle
            opt_solver.solve_sco2_case(display=False)
            solution = opt_solver.m_solve_dict
        except: 
            solution = 'Solution not found. Optimization failed to converge.'
        elapsed = clock.toc()

        tracker.value += 1
        queue.put((study, solution, elapsed))
    def _mplistener(self, queue, tracker, total, cores): 
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
                ("The cycle efficiency is", "eta_thermal_calc"),
                ("The HTF mass flow rate", "m_dot_htf_des"),
                ("The HTF dT across PHX is", "deltaT_HTF_PHX"), 
                ("The LTR UA fraction is", "recup_LTR_UA_frac"),
                ("The LTR UA is", "LTR_UA_calculated"),
                ("The HTR UA is", "HTR_UA_calculated"),
                ("The cycle cost (M$) is", "cycle_capital_cost"),
                ("The plant cost (M$) is", "plant_capital_cost"),
                ("The total cost (M$) is", "total_capital_cost"),
                ("The total adjusted cost is", "total_adjusted_cost"),
                ("The levelized cost of energy is", "levelized_cost_of_energy"),
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
            elif remaining_iters > 120: 
                message = f'{remaining_time / 60:.2f} minutes' 
            else: message = 'nearly complete'

            return message

        display = getlines()
        timeset = np.full(total, np.nan)
        message = ['Initializing parametric study...', '\n', '\n']
        while True: 
            writer(message)
            try: results = queue.get(timeout=2) 
            except Empty: 
                continue
            firstline = f' study {tracker.value:03} / {total:03} '
            if results=='kill': 
                for _ in range(len(study)+len(display)+3): sys.stdout.write('\n')
                break
            elif isinstance(results, tuple) and isinstance(results[1], dict): 
                message = [f'#-{firstline:-^41}-#']

                study    = results[0]
                solution = results[1]
                elapsed  = results[2]

                timeset[tracker.value] = elapsed
                message.append(f'\n{"Time to completion":.<30}{timeleft(tracker.value, total, timeset):.>15}')
                message.append('\n')
                for key, value in study.items(): 
                    message.append(f'\n{key:.<35}{value:.>10.4f}')

                message.append('\n')
                try: 

                    for line in display: 
                        message.append(f'\n{line[0]:.<35}{solution[line[1]]:.>10.4f}')
                    self.solution = solution
                    self.solutions.append(self.solution.copy())
                    self._savesolution()

                except KeyError: 

                    for line in display:                 
                        message.append(f'\n{line[0]:.<35}{"nan":.>10}')

                message.append('\n')
                writer(message)
            elif isinstance(results, tuple) and isinstance(results[1], str): 
                message = [f'#-{firstline:-^41}-#']
                
                study    = results[0]
                solution = results[1]
                elapsed  = results[2]

                timeset[tracker.value] = elapsed
                message.append(f'\n{"Time to completion":.<30}{timeleft(tracker.value, total, timeset):.>15}')
                message.append('\n')
                for key, value in study: 
                    message.append(f'\n{key:.>35}{value:.>10.4f}')

                message.append('\n')
                for line in display:                 
                    message.append(f'\n{line[0]:.<35}{"nan":.>10}')
                message.append('\n')
                writer(message)
    def _savesolution(self) -> None: 
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
            fields = list(self.solution.keys())
            writer = csv.DictWriter(file, fieldnames=fields)
            if not header: 
                writer.writeheader()
            writer.writerow(self.solution)
    def _defaults(self, params: dict={}) -> None: 
        '''
        The default design parameter dictionary.
        '''
        self.parameters["quiet"]                = 1         # [-]   If true (1), no status = successful log notices. 
        self.parameters["opt_logging"]          = 0         # [-]   If true (1), save each opt loop result to objective.csv.
        self.parameters["opt_penalty"]          = 0         # [-]   If true (1), allow addition of penalty terms to objective.
        self.parameters["try_simple_cycle"]     = 1         # [-]   If true (1), the optimizer will check a simple cycle after convergence. 
        
        #---System design parameters
        self.parameters["htf"]                  = 36        # [-]   See design_parameters.txt
        self.parameters["T_htf_hot_des"]        = 700.0     # [C]   HTF design hot temperature (PHX inlet)
        self.parameters["dT_PHX_hot_approach"]  = 20.0      # [C/K] Default=20. Temperature difference between hot HTF and turbine inlet
        self.parameters["T_amb_des"]            = 35.0      # [C]   Ambient temperature at design
        self.parameters["dT_mc_approach"]       = 6.0       # [C]   Use 6 here per Neises & Turchi 19. Temperature difference between main compressor CO2 inlet and ambient air
        self.parameters["site_elevation"]       = 588       # [m]   Elevation of Daggett, CA. Used to size air cooler...
        self.parameters["W_dot_net_des"]        = 100.0     # [MWe] Design cycle power output (no cooling parasitics)
        self.parameters["TES_capacity"]         = 12.0      # [hours] Thermal engery storage hours
        self.parameters["heliostat_cost"]       = 75.0      # [$/m^2] Cost per m^2 of heliostat reflective surface area.
        self.parameters["receiver_eta_mod"]     = 1.0       # [-]   Modifies the receiver efficiency. If <0, it overrides the efficiency instead.
        #---Cycle design options
            # Configuration
        self.parameters["cycle_config"]         = 1         # [-]   If [1] = RC, [2] = PC
            # Recuperator design
        self.parameters["design_method"]        = 2         # [-]   1 = specify efficiency, 2 = specify total recup UA, 3 = Specify each recup design (see inputs below)
        self.parameters["eta_thermal_des"]      = 0.44      # [-]   Target power cycle thermal efficiency (used when design_method == 1)
        self.parameters["UA_recup_tot_des"]     = 15 * 1000 * (self.parameters["W_dot_net_des"]) / 50.0  # [kW/K] (used when design_method == 2). If < 0, optimize. 
            # Pressures and recompression fraction
        self.parameters["is_recomp_ok"]         = 1 	    # [-]   1 = Yes, 0 = simple cycle only, < 0 = fix f_recomp to abs(input)
        self.parameters["is_P_high_fixed"]      = 1         # [-]   0 = No, optimize. 1 = Yes (=P_high_limit)
        self.parameters["is_PR_fixed"]          = 0         # [-]   0 = No, >0 = fixed pressure ratio at input <0 = fixed LP at abs(input)
        self.parameters["is_IP_fixed"]          = 0         # [-]   partial cooling config: 0 = No, >0 = fixed HP-IP pressure ratio at input, <0 = fixed IP at abs(input)
        #---Convergence and optimization criteria
        self.parameters["des_objective"]        = 1         # [-]   [2] = hit min phx deltat then max eta, [3] = min cost, [else] max eta
        self.parameters["rel_tol"]              = 3         # [-]   Baseline solver and optimization relative tolerance exponent (10^-rel_tol)
        # Weiland & Thimsen 2016
        # In most studies, 85% is an accepted isentropic efficiency for either the main or recompression compressors, and is the recommended assumption.
        self.parameters["eta_isen_mc"]          = 0.85      # [-]   Main compressor isentropic efficiency
        self.parameters["eta_isen_rc"]          = 0.85      # [-]   Recompressor isentropic efficiency
        self.parameters["eta_isen_pc"]          = 0.85      # [-]   Precompressor isentropic efficiency
        self.parameters["gross_to_net"]         = 0.90      # [-]   Turbine / generator gross-to-net conversion factor
        # Weiland & Thimsen 2016
        # Recommended turbine efficiencies are 90% for axial turbines above 30 MW, and 85% for radial turbines below 30 MW.
        self.parameters["eta_isen_t"]           = 0.90      # [-]   Turbine isentropic efficiency
        self.parameters["P_high_limit"]         = 25.0      # [MPa] Cycle high pressure limit
        # Weiland & Thimsen 2016
        # Multiple literature sources suggest that recuperator cold side (high pressure) pressure drop of
        # approximately 140 kPa (20 psid) and a hot side (low pressure) pressure drop of 280 kPa (40 psid) can be reasonably used.
        # Note: Unclear what the low pressure assumption is in this study, could be significantly lower for direct combustion cycles
        dP_HP_HX = 0.0056  # [-] = 0.14[MPa]/25[MPa]
        dP_LP_HX = 0.0311  # [-] = 0.28[MPa]/9[MPa]
        #---LTR
        self.parameters["LTR_design_code"]      = 3         # [-]   1 = UA, 2 = min dT, 3 = effectiveness
        self.parameters["LTR_UA_des_in"]        = 2200.0    # [kW/K] (required if LTR_design_code == 1)
        self.parameters["LTR_min_dT_des_in"]    = 12.0      # [C]   (required if LTR_design_code == 2)
        self.parameters["LTR_eff_des_in"]       = 0.895     # [-]   (required if LTR_design_code == 3)
        self.parameters["LT_recup_eff_max"]     = 1.0       # [-]   Maximum effectiveness low temperature recuperator
        self.parameters["LTR_LP_deltaP_des_in"] = dP_LP_HX  # [-]   LP-side pressure loss fraction.
        self.parameters["LTR_HP_deltaP_des_in"] = dP_HP_HX  # [-]   HP-side pressure loss fraction.
        #---HTR
        self.parameters["HTR_design_code"]      = 3         # [-]   1 = UA, 2 = min dT, 3 = effectiveness
        self.parameters["HTR_UA_des_in"]        = 2800.0    # [kW/K] (required if LTR_design_code == 1)
        self.parameters["HTR_min_dT_des_in"]    = 19.2      # [C]   (required if LTR_design_code == 2)
        self.parameters["HTR_eff_des_in"]       = 0.945     # [-]   (required if LTR_design_code == 3)
        self.parameters["HT_recup_eff_max"]     = 1.0       # [-]   Maximum effectiveness high temperature recuperator
        self.parameters["HTR_LP_deltaP_des_in"] = dP_LP_HX  # [-]   LP-side pressure loss fraction. 
        self.parameters["HTR_HP_deltaP_des_in"] = dP_HP_HX  # [-]   HP-side pressure loss fraction. 
        #---PHX
        self.parameters["PHX_co2_deltaP_des_in"] = dP_HP_HX # [-]   Heat exchanger pressure loss fraction. 
        self.parameters["dT_PHX_cold_approach"] = 20        # [C/K] Default = 20. Temperature difference between cold HTF and cold CO2 PHX inlet
        self.parameters["PHX_n_sub_hx"]         = 10        # [-]   Subdivisions used when designing PHX. 
        self.parameters["PHX_cost_model"]       = 100       # [-]   $/UA of PHX. [3] = baseline, [10] = 10% baseline, [50] = 50% baseline, etc. 
        #---Air Cooler
        self.parameters["deltaP_cooler_frac"]   = 0.005     # [-]   Fraction of CO2 inlet pressure that is design point cooler CO2 pressure drop
        self.parameters["fan_power_frac"]       = 0.02      # [-]   Fraction of net cycle power consumed by air cooler fan. 2% here per Turchi et al.
        #---Default
        self.parameters["deltaP_counterHX_frac"] = 0.0054321 # [-] Fraction of CO2 inlet pressure that is design point counterflow HX (recups & PHX) pressure drop
        self.parameters.update(params)

if __name__=='__main__': 

    print("current processID:", os.getpid(), "\n") 

    system = System(newrun=True, file='solutions.csv')
    system.update({
        'try_simple_cycle'     : 1,      # [-]
        'receiver_eta_mod'     : 1.0,    # [-]
        'heliostat_cost'       : 75,     # [$/m^2]
        'T_htf_hot_des'        : 700,    # [C] 
        'opt_penalty'          : 1,      # [-] 
        'opt_logging'          : 0,      # [-] 
        'LTR_min_dT_des_in'    : 5,      # [C] 
        'HTR_min_dT_des_in'    : 5,      # [C] 
        'dT_PHX_hot_approach'  : 280.0,  # [C] 
        'dT_PHX_cold_approach' : 20.0,   # [C] 
        'PHX_cost_model'       : 100.0,  # [-]
        'gross_to_net'         : 0.98,   # [-]
        'W_dot_net_des'        : 100.0,  # [MWe]
    })

    # system.update(prebuilt='maxeta')
    # system.optimize(savefig=True)

    # system.update(prebuilt='minlcoe')
    # system.optimize(savefig=True)

    system.update(prebuilt='minlcoe')
    system.parametric(params={
        'W_dot_net_des'       : [100], 
        'heliostat_cost'      : np.arange(50, 100, 5),  
        'PHX_cost_model'      : np.arange(60, 1001, 20), 
        'dT_PHX_hot_approach' : np.arange(100, 301, 10), 
        'dT_PHX_cold_approach': np.arange(20, 101, 20), 
        'T_htf_hot_des'       : np.arange(700, 1100, 20)
    }, cores=4)



