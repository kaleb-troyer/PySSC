import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pyfluids import Fluid, FluidsList, Input
from itertools import product

path = 'Brayton Cycle/co2_tables'
temp_kelvin = False

SR   = {'min': 0.2, 'max': 0.8}
PR   = {'min': 3.3, 'max': 3.3}
RPR  = {'min': 0.2, 'max': 0.8}
Pmax = {'min':  25, 'max':  25} # MPa
Pmid = {'min':  10, 'max':  18} # MPa (unused)
CIT  = {'min':  50, 'max':  50} # C
TIT  = {'min': 800, 'max': 800} # C

optimization_iter = 3

cycles_to_test = {
    "simple cycle" : False,
    "reheat cycle" : False,
    "recoup cycle" : True,
    "recoup reheat": False, 
}

# data collection and parsing functions, optimization
def state_finder(pressure, match_value, match_name): # returns a row from the dataframe, given one value in that row

    sCO2 = Fluid(FluidsList.CarbonDioxide) 
    if match_name == 'temp.':
        sCO2.update(Input.pressure(pressure), Input.temperature(match_value))
    elif match_name == 'entr.':
        sCO2.update(Input.pressure(pressure), Input.entropy(match_value*1000))
    elif match_name == 'enth.':
        sCO2.update(Input.pressure(pressure), Input.enthalpy(match_value*1000))

    column_names = {'temperature': 'temp.', 'pressure': 'pres.', 'density': 'dens.', 'specific_volume': 'volu.', 'enthalpy': 'enth.', 'entropy': 'entr.', 'specific_heat': 'Cp'}
    row = pd.Series(sCO2.as_dict()).rename(column_names)

    row['enth.'] = row['enth.'] / 1000
    row['entr.'] = row['entr.'] / 1000
    row['Cp'] = row['Cp'] / 1000

    return row
def table_builder(pressure, path=path): # builds co2 isobaric dataframe given pressure
    def data_loader(pressure, path=path):
        return pd.read_csv(path+'/'+'{:02}'.format(int(pressure))+'mpa.txt', delimiter='\t').drop(columns='Phase')
    
    pressure = round(pressure / 1e6, 2)
    if pressure-int(pressure)==0 and pressure >= 0:
        isobar_table_C = data_loader(pressure)
    elif pressure > 0:
        pressure_A = np.floor(pressure)
        pressure_B = np.ceil(pressure)
        fraction = (pressure-pressure_A)/(pressure_B-pressure_A)
        isobar_table_A = data_loader(pressure_A)
        isobar_table_B = data_loader(pressure_B)

        isobar_table_C = isobar_table_A + (fraction * (isobar_table_B - isobar_table_A))

    isobar_table_C.columns = ['temp.', 'pres.', 'dens.', 'volu.', 'internal energy', 'enth.', 'entr.', 'Cv', 'Cp', 'speed of sound', 'joule-thomson', 'viscosity', 'conductivity']
    return isobar_table_C
def get_precise_range(df=pd.DataFrame(), high=0, low=0, column_name='temp.'): # truncates isobaric table given high and low values, eg. temperature
    specific_range = df[(df[column_name] > low) & (df[column_name] < high)]
    return specific_range
def get_precise_table(df=pd.DataFrame(), high=0, low=0, column_name='temp.'):
    if high < low:
        high, low = low, high
    tableRange = get_precise_range(df=df, high=high, low=low, column_name=column_name)
    return tableRange
def get_precise_value(pressure, temperature, path=path):
    """
    Finds the enthalpy value corresponding to a given pressure and temperature.

    Parameters:
        pressure (float): The pressure in Pascals.
        temperature (float): The temperature in degrees Celsius.
        path (str): Path to the directory containing isobaric data files.

    Returns:
        float: The enthalpy value ('enth.') corresponding to the input pressure and temperature.
    """
    # Build the isobaric table for the given pressure
    isobar_table = table_builder(pressure, path=path)
    
    # Use get_precise_range to find the closest range in temperature
    precise_range = get_precise_table(df=isobar_table, high=temperature+1, low=temperature-1, column_name='temp.')
    
    if precise_range.empty:
        raise ValueError(f"No data found for pressure {pressure} and temperature {temperature}.")
    
    # Interpolate to find the precise enthalpy value at the given temperature
    enthalpy = np.interp(
        temperature,
        precise_range['temp.'].values,
        precise_range['enth.'].values
    )
    
    return enthalpy
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 2, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
def optimizer(function):
    def wrapper(**kwargs):
        # parsing parameters and ranges for optimization
        parameters = {key: value for key, value in kwargs.items()}
        results_list = []
        initial_parameters = parameters.copy()
        override_enabler   = False
        step_size_modifier = 0
        best_cycle = {'cycle': None, 'efficiency': float('-inf'), 'params': None}
        next_best  = {'cycle': None, 'efficiency': float('-inf'), 'params': None}
        for k in range(optimization_iter):
            parameter_ranges = {}
            step_size_modifier += 0.5
            for key, value in parameters.items():
                step_size = (value['max'] - value['min']) / (10 * step_size_modifier)
                if step_size != 0: 
                    parameter_ranges[key] = np.arange(value['min'], value['max']+(step_size/2), step_size)
                else: parameter_ranges[key] = np.array([value['min']]) # if step = 0
            

            all_combinations = list(product(*parameter_ranges.values()))
            total_iterations = len(all_combinations)
            for i, row in enumerate(all_combinations):
                # tracking optimization progress
                printProgressBar(i+1, total_iterations, length=22, prefix=f' Iteration {k+1}/{optimization_iter}:')
                # passing current iteration parameters to specific cycle function
                current_combination = {}
                for j, key in enumerate(parameters.keys()):
                    current_combination[key] = round(row[j], 3)
                results = function(**current_combination)
                results_list.append(results)
                # breaking current test if results are invalid
                if results == False: continue
                # if new best cycle is found, next best is set to old value
                elif results['efficiency'] > best_cycle['efficiency']: 
                    next_best = best_cycle
                    best_cycle = results
                # if no new best is found, but results are greater than next best
                elif results['efficiency'] > next_best['efficiency']:
                    next_best = results
            if next_best['cycle'] == None: break
            for key, value in best_cycle['params'].items():
                if parameters[key]['min'] == parameters[key]['max']: continue
                # deciding whether to iterate towards max allowed or next best
                if value == parameters[key]['max'] and override_enabler==True:
                    next_best['params'][key] = initial_parameters[key]['max']
                # defining next iteration min and max values
                if next_best['params'][key] > value:
                    parameters[key] = {'min': value, 'max': next_best['params'][key]}
                elif next_best['params'][key] == value:
                    if value == initial_parameters[key]['min']:
                        parameters[key] = {'min': value, 'max': value}
                    else: parameters[key] = {'min': value, 'max': initial_parameters[key]['max']}
                else: parameters[key] = {'min': next_best['params'][key], 'max': value}
            # override enabler helps decide wether to explore left or right side of plot
            override_enabler = True

        # checking for valid solution
        if best_cycle['cycle']==None: raise ValueError('No valid solution found. Cycle = None.')
        
        # sending results to csv
        file_name = f"Brayton Cycle/results/{best_cycle['cycle'].label}.csv"  
        with open(file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header if the file is empty
            header = ['efficiency'] + list(kwargs.keys())
            csv_writer.writerow(header)

            # Write the row
            for result in results_list:
                if result == False: continue
                efficiency = result['efficiency']
                kwargs = result['params']

                # Construct the row
                row = [efficiency] + list(kwargs.values())
                csv_writer.writerow(row)
        return best_cycle
    return wrapper

if not temp_kelvin: # converting celsius to kelvin if necessary
    for key, value in CIT.items(): 
        if key != 'step': CIT[key] += 273.15
    for key, value in TIT.items(): 
        if key != 'step': TIT[key] += 273.15
for key in Pmax.keys(): # converting MPa to Pa
    Pmax[key] = Pmax[key] * 1e6
    Pmid[key] = Pmid[key] * 1e6

# defining equipment parent class and cycle components
class Equipment():
    def __init__(self, label='None', cycle=None, massFlowPercent=1):
        # defining basic equipment information
        self.label = label
        self.cycle = cycle
        self.stage = self.cycle.stage
    def work_produced(self):
        if type(self) == Recuperator:
            self.work = self.heat_transferred
        else: self.work = self.massFlowPercent * (self.inlet_entha - self.leave_entha)
        return self.work  # [kJ/kg]
class Heater(Equipment): 
    def __init__(self, label='None', cycle=None, P_delta=0, P_start=None, T_start=None, T_final=0, efficiency=1, massFlowPercent=1):
        # initializing parent class
        super().__init__(label=label, cycle=cycle)
        self.__dict__.update(locals())
        self._update_inlet_and_outlet(self.cycle.state)
    def _update_inlet_and_outlet(self, inlet=pd.Series()):
        # instantiating inlet state and adiabatic outlet state
        if inlet is not None: 
            self.inlet_state = inlet
            self.pressure_hi = self.inlet_state['pres.']
        elif self.P_start != None and self.T_start != None:
            self.pressure_hi = self.P_start
            self.inlet_state = state_finder(self.pressure_hi, self.T_start, 'temp.')
        else: raise ValueError('Must specify an initial pressure and temperature for first equipment class. P_start=None, T_start=None.')
        self.inlet_entha = self.inlet_state['enth.']
        self.pressure_lo = self.inlet_state['pres.'] - self.P_delta
        self.leave_state = state_finder(self.pressure_lo, self.T_final, 'temp.')

        # modifying outlet state based on process efficiency
        self.leave_entha = self.inlet_entha+((self.leave_state['enth.']-self.inlet_entha)/self.efficiency)
        self.leave_state = state_finder(self.pressure_lo, self.leave_entha, 'enth.')
    def costing(self, mass_flow_rate):
        pass # calculate cost
class Cooler(Equipment): 
    def __init__(self, label='None', cycle=None, P_delta=0, P_start=None, T_start=None, T_final=0, efficiency=1, massFlowPercent=1):
        # initializing parent class
        super().__init__(label=label, cycle=cycle)
        self.__dict__.update(locals())
        self._update_inlet_and_outlet(self.cycle.state)
    def _update_inlet_and_outlet(self, inlet=pd.Series()):
        # instantiating inlet state and adiabatic outlet state
        if inlet is not None: 
            self.inlet_state = inlet
            self.pressure_hi = self.inlet_state['pres.']
        elif self.P_start != None and self.T_start != None:
            self.pressure_hi = self.P_start
            self.inlet_state = state_finder(self.pressure_hi, self.T_start, 'temp.')
        else: raise ValueError('Must specify an initial pressure and temperature for first equipment class. P_start=None, T_start=None.')
        self.inlet_entha = self.inlet_state['enth.']
        self.pressure_lo = self.inlet_state['pres.'] - self.P_delta
        self.leave_state = state_finder(self.pressure_lo, self.T_final, 'temp.')

        # modifying outlet state based on process efficiency
        self.leave_entha = self.inlet_entha+((self.leave_state['enth.']-self.inlet_entha)/self.efficiency)
        self.leave_state = state_finder(self.pressure_lo, self.leave_entha, 'enth.')
    def costing(self, mass_flow_rate):
        pass # calculate cost
class Turbine(Equipment): # expansion to specified pressure 
    def __init__(self, label='None', cycle=None, P_final='Pmin', P_start=None, T_start=TIT, efficiency=1, massFlowPercent=1):
        # initializing parent class
        super().__init__(label=label, cycle=cycle)
        self.__dict__.update(locals())
        self._update_inlet_and_outlet(self.cycle.state)
    def _update_inlet_and_outlet(self, inlet=pd.Series()):
        # instantiating inlet state and adiabatic outlet state
        if inlet is not None: 
            self.inlet_state = inlet
            self.pressure_hi = self.inlet_state['pres.']
        elif self.P_start != None:
            self.pressure_hi = self.P_start
            self.inlet_state = state_finder(self.pressure_hi, self.T_start, 'temp.')
        else: raise ValueError('Must specify an initial pressure for first equipment class. P_start=None.')
        self.inlet_entro = self.inlet_state['entr.']
        self.inlet_entha = self.inlet_state['enth.']
        self.pressure_lo = self.P_final
        self.leave_state = state_finder(self.pressure_lo, self.inlet_entro, 'entr.')

        # modifying outlet state based on process efficiency
        self.leave_entha = self.inlet_entha+((self.leave_state['enth.']-self.inlet_entha)*self.efficiency)
        self.leave_state = state_finder(self.pressure_lo, self.leave_entha, 'enth.')
    def costing(self, massFlowRate):
        pass # calculate cost
class Compressor(Equipment): 
    def __init__(self, label='None', cycle=None, P_final='Pmax', P_start=None, T_start=CIT, efficiency=1, massFlowPercent=1):
        # initializing parent class
        super().__init__(label=label, cycle=cycle)
        self.__dict__.update(locals())
        self._update_inlet_and_outlet(self.cycle.state)
    def _update_inlet_and_outlet(self, inlet=pd.Series()):
        # instantiating inlet state and adiabatic outlet state
        if inlet is not None: 
            self.inlet_state = inlet
            self.pressure_lo = self.inlet_state['pres.']
        elif self.P_start != None:
            self.pressure_lo = self.P_start
            self.inlet_state = state_finder(self.pressure_lo, self.T_start, 'temp.')
        else: raise ValueError('Must specify an initial pressure for first equipment class. P_start=None.')
        self.inlet_entro = self.inlet_state['entr.']
        self.inlet_entha = self.inlet_state['enth.']
        self.pressure_hi = self.P_final
        self.leave_state = state_finder(self.pressure_hi, self.inlet_entro, 'entr.')

        # modifying outlet state based on process efficiency
        self.leave_entha = self.inlet_entha+((self.leave_state['enth.']-self.inlet_entha)/self.efficiency)
        self.leave_state = state_finder(self.pressure_hi, self.leave_entha, 'enth.')
    def costing(self, massFlowRate):
        pass # calculate cost
class Recuperator(Equipment): 
    def __init__(self, 
                 label='None',          # name of recuperator
                 cycle=None,            # cycle to add recuperator to
                 P_delta=0,             # pressure loss across recuperator
                 coolerName=None,       # cold HX to connect recuperator to
                 heaterName=None,       # warm HX to connect recuperator to
                 T_warm_margin=None,    # enforced temperature difference on warm side
                 T_cold_margin=None,    # enforced temperature difference on cold side
                 splitRatio=1,          # ratio of mass flow from warm side to cold side
                 efficiency=1):
        # initializing parent class
        super().__init__(label=label, cycle=cycle)
        self.__dict__.update(locals())
        self._errorHandling()
        self._outletStateFinder()
        self.states = {f'{self.label}, cold side': [self.cold_inlet_state, self.cold_leave_state],
                       f'{self.label}, warm side': [self.warm_inlet_state, self.warm_leave_state]}
        # self._physicalityEval()
        # self.plot_exchange()
    def _errorHandling(self):
        # confirming all necessary information has been provided
        if self.coolerName==None or type(self.heaterName)==None: 
            raise ValueError('Must specify heater and cooler to be modified by recuperator.')
        elif self.T_warm_margin==None and self.T_cold_margin==None:
            raise ValueError('Must specify a warm outlet temperature OR a cold outlet temperature.')
        elif self.T_warm_margin!=None and self.T_cold_margin!=None: #LMTD
            raise ValueError('Due to non-constant Cp values, LMTD may not be used. Pick only one outlet temperature.')
        else: pass
    def _outletStateFinder(self):
        # locating heater and cooler based on provided names
        self.heater = self.cycle.cycle_equipment[self.heaterName]['classObject']
        self.cooler = self.cycle.cycle_equipment[self.coolerName]['classObject']

        # defining inlet states based on heater and cooler
        self.warm_inlet_state = self.cooler.inlet_state
        self.cold_inlet_state = self.heater.inlet_state

        # getting isobaric pressure tables for inlets and outlets, including pressure loss
        self.warm_inlet_pressure = self.warm_inlet_state['pres.']
        self.warm_leave_pressure = self.warm_inlet_pressure - self.P_delta
        self.cold_inlet_pressure = self.cold_inlet_state['pres.']
        self.cold_leave_pressure = self.cold_inlet_pressure - self.P_delta

        # if guessing warm outlet temperature
        if self.T_warm_margin != None: 
            # calculating outlet states according to outlet temperature guess and delta-enthalpy
            self.temp_leave_guess = self.T_warm_margin + self.heater.inlet_state['temp.']
            self.warm_leave_state = state_finder(self.warm_leave_pressure, self.temp_leave_guess, 'temp.')
            self.heat_transferred = self.cooler.massFlowPercent * (self.warm_inlet_state['enth.']-self.warm_leave_state['enth.'])
            self.cold_leave_entha = (self.heat_transferred/self.heater.massFlowPercent) + self.cold_inlet_state['enth.']
            self.cold_leave_state = state_finder(self.cold_leave_pressure, self.cold_leave_entha, 'enth.')

        # if guessing cold outlet temperature
        elif self.T_cold_margin != None: 
            # calculating outlet states according to outlet temperature guess and delta-enthalpy
            self.temp_leave_guess = self.T_cold_margin + self.cooler.inlet_state['temp.']
            self.cold_leave_state = state_finder(self.cold_leave_pressure, self.temp_leave_guess, 'temp.')
            self.heat_transferred = self.heater.massFlowPercent * (self.cold_inlet_state['enth.']-self.cold_leave_state['enth.'])
            self.warm_leave_entha = (self.heat_transferred/self.cooler.massFlowPercent) + self.warm_inlet_state['enth.']
            self.warm_leave_state = state_finder(self.warm_leave_pressure, self.warm_leave_entha, 'enth.')
    def _physicalityEval(self, subdivisions=100):
        self.mf_c = self.heater.massFlowPercent
        self.mf_h = self.cooler.massFlowPercent
        self.T_hi = self.warm_inlet_state['temp.']
        self.T_ci = self.cold_inlet_state['temp.']

        T_ho = self.warm_leave_state['temp.']
        self.subdivisions = subdivisions

        cold_inlet = state_finder(self.cold_inlet_pressure, self.T_ci, 'temp.')
        warm_inlet = state_finder(self.warm_inlet_pressure, self.T_hi, 'temp.')
        warm_leave = state_finder(self.warm_leave_pressure, T_ho, 'temp.')

        h_hi = warm_inlet['enth.']
        h_ho = warm_leave['enth.']

        q_total = self.mf_h * (h_hi - h_ho)
        q_i = q_total / subdivisions
        i = 0

        row_1 = {
            "q_i"   : q_i,

            "T_ih"  : self.T_hi,
            "h_ih"  : h_hi,
            "mf_h"  : self.mf_h,
            "cp_h"  : warm_inlet['Cp'],

            "T_ic"  : 0,
            "h_ic"  : 0,
            "mf_c"  : self.mf_c,
            "cp_c"  : 0,
        }
        subdivision_points = pd.DataFrame([row_1])
        while i < subdivisions:
            i += 1

            h_ih = subdivision_points.at[i-1, 'h_ih'] - (q_i / self.mf_h)
            warm_state_i = state_finder(self.warm_inlet_pressure, h_ih, 'enth.')
            row_i = {
                "q_i"   : q_i,

                "T_ih"  : warm_state_i['temp.'],
                "h_ih"  : h_ih,
                "mf_h"  : self.mf_h,
                "cp_h"  : warm_state_i['Cp'],

                "T_ic"  : 0,
                "h_ic"  : 0,
                "mf_c"  : self.mf_c,
                "cp_c"  : 0,
            }

            subdivision_points = pd.concat([subdivision_points, pd.DataFrame([row_i])], ignore_index=True)

        subdivision_points.at[i, 'T_ic'] = self.T_ci
        subdivision_points.at[i, 'h_ic'] = cold_inlet['enth.']
        subdivision_points.at[i, 'cp_c'] = cold_inlet['Cp']
        while i > 0: 
            i -= 1

            h_ic = subdivision_points.at[i+1, 'h_ic'] + (q_i / self.mf_c)
            cold_state_i = state_finder(self.cold_inlet_pressure, h_ic, 'enth.')

            subdivision_points.at[i, 'h_ic'] = h_ic
            subdivision_points.at[i, 'T_ic'] = cold_state_i['temp.']
            subdivision_points.at[i, 'cp_c'] = cold_state_i['Cp']

        subexchanger_properties = {}
        for i in range(subdivisions):
            this_row = subdivision_points.loc[i]
            next_row = subdivision_points.loc[i+1]
            Cmin = min(self.mf_c * (this_row['cp_c']+next_row['cp_c'])/2, self.mf_h * (this_row['cp_h']+next_row['cp_h'])/2)
            warm_temp_inlet = this_row['T_ih']
            cold_temp_inlet = next_row['T_ic']
            local_effectivenss = q_i / (Cmin * (warm_temp_inlet - cold_temp_inlet))
            subexchanger_properties[f'{i}-{i+1}'] = local_effectivenss
            if local_effectivenss > 1: 
                self.nonphysical = True
                return None
        
        self.warm = subdivision_points['T_ih'].to_numpy()
        self.cold = subdivision_points['T_ic'].to_numpy()
        self.effectivenesses = np.array(list(subexchanger_properties.values()))
        self.average_effectiveness = self.effectivenesses.mean()
    def plot_exchange(self):
        plt.figure()
        plt.plot(self.warm, label=f'warm side ({self.warm_inlet_state["pres."]:.2f} MPa)', color='red')
        plt.plot(self.cold, label=f'cold side ({self.cold_inlet_state["pres."]:.2f} MPa)', color='blue')
        plt.plot(max(self.warm), label='effectiveness', linestyle='--', color='#435f75')
        plt.xlabel('subdivision points')
        plt.ylabel('Temperature (K)')
        plt.annotate(f'SR = {self.mf_c/self.mf_h:.1%}', (-1, min(self.warm)+20))
        plt.legend()

        effectiveness_plot = plt.gca().twinx()
        effectiveness_plot.plot(self.effectivenesses, label='effectiveness', linestyle='--', color='#435f75')
        effectiveness_plot.set_ylim(bottom=0, top=1)
        effectiveness_plot.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        effectiveness_plot.set_ylabel('Effectiveness')
        plt.title(f'sCO2 Recuperator Model | {self.subdivisions} Subdivisions')
        plt.tight_layout()
        plt.show()
    def costing(self, mass_flow_rate):
        pass # calculate cost

# defining cycle class and functions
class Cycle():
    def __init__(self, label='None', **kwargs):
        # defining vars for tracking current stage
        self.label = label
        self.stage = 1
        self.state = None
        self.work_produced = 0
        self.heat_supplied = 0

        # this dictionary tracks name, inlet and outlet states, and equipment class object
        self.cycle_equipment = {}
        self.orderedKeysList = []

        # specifying pressure ranges, eg. Pmin, Pmax, etc.
        self.pressures = {}
        for key, value in kwargs.items():
            self.pressures[key] = {'pressure': value, 'table': np.NaN}
    # user-end master functions for adding equipment to cycle or getting info
    def add_equipment(self, type='None', label='None', **kwargs):
        # types of equipment allowed
        equipment_directory = {
            'recuperation': self._addRecuperator, 
            'flowdivision': self._addFlowDivider,
            'compression':  self._addCompression,
            'expansion':    self._addExpansion,
            'heating':      self._addHeater, 
            'cooling':      self._addCooler,
        }

        # error handling for equipment type
        if type not in equipment_directory.keys():
            raise ValueError('Equipment type not recognized.')

        # equipment instance created
        instance = equipment_directory[type](label=label, **kwargs)
        self._cycleEquipmentUpdater(instance)
        self._calculate_efficiency()
    def plot_cycle(self, figure_info=None):
        def darken_color(color_hex, factor):
            # Extract individual RGB components
            red = int(color_hex[1:3], 16)
            green = int(color_hex[3:5], 16)
            blue = int(color_hex[5:7], 16)

            # Darken the color
            darkened_red = int(red * (1 - factor))
            darkened_green = int(green * (1 - factor))
            darkened_blue = int(blue * (1 - factor))

            # Convert back to hex format
            darkened_color_hex = f"#{darkened_red:02x}{darkened_green:02x}{darkened_blue:02x}"
            return darkened_color_hex
        color = '#3BB143'
        initial_color = color
        
        # building figure from figure info
        row = figure_info['rcount']
        col = figure_info['ccount']
        plt.subplot(row, col, figure_info['subplt'])

        # iterating to plot each process
        for item in self.orderedKeysList:
            equipment = self.cycle_equipment[item]
            start = equipment['stage_start']
            close = equipment['stage_close']
            table = table_builder(close['state']['pres.'])
            label = f"({start['point']}-{close['point']}) {item}"

            x_start = start['state']['entr.']
            x_close = close['state']['entr.']
            y_start = start['state']['temp.']
            y_close = close['state']['temp.']

            # identifying process (isobaric or adiabatic)
            if type(equipment['classObject']) in (Recuperator, Heater, Cooler):
                precise_table = get_precise_table(table, high=y_close, low=y_start)
                x_values = precise_table['entr.'].to_numpy()
                y_values = precise_table['temp.'].to_numpy()
            else: 
                x_values = [x_start, x_close]
                y_values = [y_start, y_close]

            # plotting data
            plt.plot(x_values, y_values, label=label, color=color)
            plt.plot(x_start, y_start, marker='o', markersize=3, color=color)
            plt.annotate(str(start['point']), (x_start, y_start))    
            color = darken_color(color, 0.1)
        
        # plotting final point and building figure
        plt.plot(self.initial_state['entr.'], self.initial_state['temp.'], marker='o', markersize=3, color=initial_color)
        plt.title(self.label)
        plt.annotate(f'η = {self.cycle_efficiency:0.2%}  ', (plt.xlim()[1], plt.ylim()[0]+20), ha='right')
        plt.xlabel('s (kJ/kg * K)')
        plt.ylabel('T (K)')
        if figure_info['curcol'] > 1:
            plt.ylabel('')
            plt.gca().tick_params(left=False, labelleft=False)
        plt.legend()

        figure_info['subplt'] += 1
        if figure_info['curcol'] < figure_info['ccount']:
            figure_info['curcol'] += 1
        else: figure_info['curcol'] = 1
        for equipment in self.cycle_equipment.values():
            if type(equipment) == Recuperator:
                equipment.plot_exchange()
    def cycle_info(self):
        print(f"\n Cycle Equipment Details: {'η'} = {self._calculate_efficiency():2.2%}")
        print(70*"-")
        for key in self.orderedKeysList:
            details = self.cycle_equipment[key]
            T_start, T_leave = details['stage_start']['state']['temp.'], details['stage_close']['state']['temp.']
            P_start, P_leave = details['stage_start']['state']['pres.'], details['stage_close']['state']['pres.']
            s_start, s_leave = details['stage_start']['state']['entr.'], details['stage_close']['state']['entr.']
            print(f" Label: {key}")
            print(f" Stage Start: Point {details['stage_start']['point']}, State - T={T_start:05.2f}(K), S={s_start:04.2f}(J/g*K), P={P_start/1e6:04.1f}(MPa)")
            print(f" Stage Close: Point {details['stage_close']['point']}, State - T={T_leave:05.2f}(K), S={s_leave:04.2f}(J/g*K), P={P_leave/1e6:04.1f}(MPa)")
            print(f" Class Object: {details['classObject']}")
            print(70*"-")
        print(f"\n Work produced per cycle: {round(self.work_produced, 2)} [kJ/kg * (1) kg/s]")
    # hidden functions for creating and tracking equipment through the add_equipment function
    def _addCompression(self, label='None', **kwargs):
        instance = Compressor(label=label, cycle=self, **kwargs)
        self.work_produced += instance.work_produced()
        return instance
    def _addRecuperator(self, label='None', **kwargs):
        instance = Recuperator(label=label, cycle=self, **kwargs)
        self.heat_supplied -= instance.work_produced()
        return instance
    def _addFlowDivider(self, label='None', **kwargs):
        print('Flow divider not yet ready! \n')
        quit()
        return None
    def _addExpansion(self, label='None', **kwargs):
        instance = Turbine(label=label, cycle=self, **kwargs)
        self.work_produced += instance.work_produced()
        return instance
    def _addHeater(self, label='None', **kwargs):
        instance = Heater(label=label, cycle=self, **kwargs)
        self.heat_supplied -= instance.work_produced()
        return instance
    def _addCooler(self, label='None', **kwargs):
        instance = Cooler(label=label, cycle=self, **kwargs)
        return instance
    # for accessing pressures, efficiency, and equipment
    def _calculate_efficiency(self):
        if self.heat_supplied == 0: return 0
        self.cycle_efficiency = self.work_produced / self.heat_supplied
        return self.cycle_efficiency
    def _cycleEquipmentUpdater(self, instance=None):
        # called in add_equipment function to update all cycle equipment params
        if self.state is None: self.initial_state = instance.inlet_state

        # cycle stage and status updated
        if type(instance) != Recuperator:
            last_stage = self.stage
            equal_temp = round(instance.leave_state['temp.'], 2) == round(self.initial_state['temp.'], 2)
            equal_pres = round(instance.leave_state['enth.'], 2) == round(self.initial_state['enth.'], 2)
            if equal_temp and equal_pres:
                this_stage = 1
            else: this_stage = self.stage + 1
            self.stage = self.stage + 1
            self.state = instance.leave_state
            self.orderedKeysList.append(instance.label)
            self.cycle_equipment[instance.label] = {
                'stage_start': {'point': last_stage, 'state': instance.inlet_state},
                'stage_close': {'point': this_stage, 'state': instance.leave_state},
                'classObject': instance 
            }
        
        # recuperator requires updating equipment sequencing
        elif type(instance) == Recuperator:
            # getting info from relevant heater and cooler
            heater_info_dict = self.cycle_equipment[instance.heater.label]
            cooler_info_dict = self.cycle_equipment[instance.cooler.label] 
            warm_stage_inlet = cooler_info_dict['stage_start']['point']
            warm_stage_leave = cooler_info_dict['stage_close']['point']
            cold_stage_inlet = heater_info_dict['stage_start']['point']
            cold_stage_leave = heater_info_dict['stage_close']['point']

            # cycle equipment updated for warm / cold sides
            for key, value in instance.states.items():
                if 'cold' in key:
                    heater_info_dict['stage_start']['state'] = value[1]
                    heater_info_dict['classObject']._update_inlet_and_outlet(value[1])
                    last_stage = cold_stage_inlet
                    this_stage = cold_stage_leave
                else: # if warm side
                    cooler_info_dict['stage_start']['state'] = value[1]
                    cooler_info_dict['classObject']._update_inlet_and_outlet(value[1])
                    last_stage = warm_stage_inlet
                    this_stage = warm_stage_inlet + 1
                
                self.cycle_equipment[key] = {
                    'stage_start': {'point': last_stage, 'state': value[0]},
                    'stage_close': {'point': this_stage, 'state': value[1]},
                    'classObject': instance 
                }

            # inserting recuperator into list at appropriate index
            self.orderedKeysList.insert(self.orderedKeysList.index(instance.heater.label), list(instance.states.keys())[0])
            self.orderedKeysList.insert(self.orderedKeysList.index(instance.cooler.label), list(instance.states.keys())[1])
            cold_side_label = list(instance.states.keys())[0]
            warm_side_label = list(instance.states.keys())[1]
            cold_side_index = self.orderedKeysList.index(cold_side_label)
            warm_side_index = self.orderedKeysList.index(warm_side_label)

            # renumbering start and close points based on recuperator insertion
            for equipment in self.orderedKeysList[cold_side_index+1:]:
                equipment_info = self.cycle_equipment[equipment]
                equipment_info['stage_start']['point'] += 1
                equipment_info['stage_close']['point'] += 1
            for equipment in self.orderedKeysList[warm_side_index+1:]:
                equipment_info = self.cycle_equipment[equipment]
                equipment_info['stage_start']['point'] += 1
                equipment_info['stage_close']['point'] += 1
            self.cycle_equipment[self.orderedKeysList[-1]]['stage_close']['point'] = 1

# functions for creating specific instances of the cycle class
@optimizer
def simple_cycle(**kwargs):

    CIT  = kwargs['CIT']
    TIT  = kwargs['TIT']
    Pmax = kwargs['Pmax']
    PR   = kwargs['PR']

    Pmin = Pmax / PR

    # limiting temp and pressure to supercritical range
    if CIT < 304.128 or Pmin < 7.3773:
        return False

    cycle = Cycle(label='simple brayton cycle', Pmin=Pmin, Pmax=Pmax)

    cycle.add_equipment(type='compression', label='Main Compressor', T_start=CIT, P_start=Pmin, P_final=Pmax, efficiency=0.9)
    cycle.add_equipment(type='heating', label='Main Heater', T_final=TIT)
    cycle.add_equipment(type='expansion', label='Main Turbine', P_final=Pmin, efficiency=0.9)
    cycle.add_equipment(type='cooling', label='Main Cooler', T_final=CIT)

    return {'cycle': cycle, 'efficiency': cycle._calculate_efficiency(), 'params': kwargs}
@optimizer
def simple_cycle_w_reheat(**kwargs):

    CIT  = kwargs['CIT']
    TIT  = kwargs['TIT']
    Pmax = kwargs['Pmax']
    PR   = kwargs['PR']

    Pmin = Pmax / PR
    if 'Pmid'not in kwargs.keys():
        Pmid = ((Pmax-Pmin) / 2) + Pmin
    else: Pmid = kwargs['Pmid']
    # limiting temp and pressure to supercritical range
    if CIT < 304.128 or Pmin < 7.3773:
        return False

    cycle = Cycle(label='simple brayton cycle w/ reheat', Pmin=Pmin, Pmax=Pmax, Pmid=Pmid)

    cycle.add_equipment(type='compression', label='Main Compressor', T_start=CIT, P_start=Pmin, P_final=Pmax, efficiency=0.9)
    cycle.add_equipment(type='heating', label='Main Heater', T_final=TIT)
    cycle.add_equipment(type='expansion', label='HPT', P_final=Pmid, efficiency=0.9)
    cycle.add_equipment(type='heating', label='Reheat', T_final=TIT)
    cycle.add_equipment(type='expansion', label='LPT', P_final=Pmin, efficiency=0.9)
    cycle.add_equipment(type='cooling', label='Main Cooler', T_final=CIT)

    return {'cycle': cycle, 'efficiency': cycle._calculate_efficiency(), 'params': kwargs}
@optimizer
def simple_cycle_w_recoup(**kwargs):

    CIT  = kwargs['CIT']
    TIT  = kwargs['TIT']
    Pmax = kwargs['Pmax']
    PR   = kwargs['PR']

    Pmin = Pmax / PR
    # limiting temp and pressure to supercritical range
    if CIT < 304.128 or Pmin < 7.3773:
        return False

    cycle = Cycle(label='simple brayton cycle with recuperator', Pmin=Pmin, Pmax=Pmax)

    cycle.add_equipment(type='compression', label='Main Compressor', T_start=CIT, P_start=Pmin, P_final=Pmax, efficiency=0.9)
    cycle.add_equipment(type='heating', label='Main Heater', T_final=TIT)
    cycle.add_equipment(type='expansion', label='Main Turbine', P_final=Pmin, efficiency=0.9)
    cycle.add_equipment(type='cooling', label='Main Cooler', T_final=CIT)
    cycle.add_equipment(type='recuperation', 
                        label='Recuperator',
                        coolerName='Main Cooler',
                        heaterName='Main Heater',
                        T_warm_margin=5)

    return {'cycle': cycle, 'efficiency': cycle._calculate_efficiency(), 'params': kwargs}
@optimizer
def recoup_cycle_w_reheat(**kwargs):

    CIT  = kwargs['CIT']
    TIT  = kwargs['TIT']
    Pmax = kwargs['Pmax']
    PR   = kwargs['PR']

    Pmin = Pmax / PR
    if 'Pmid' not in kwargs.keys():
        Pmid = (((Pmax-Pmin) / 2) + Pmin)
    else: Pmid = kwargs['Pmid']
    # limiting temp and pressure to supercritical range
    if CIT < 304.128 or Pmin < 7.3773:
        return False

    cycle = Cycle(label='recuperating brayton cycle with reheat', Pmin=Pmin, Pmax=Pmax, Pmid=Pmid)

    cycle.add_equipment(type='compression', label='Main Compressor', T_start=CIT, P_start=Pmin, P_final=Pmax, efficiency=0.9)
    cycle.add_equipment(type='heating', label='Main Heater', T_final=TIT)
    cycle.add_equipment(type='expansion', label='High Pressure Turbine', P_final=Pmid, efficiency=0.9)
    cycle.add_equipment(type='heating', label='Reheat', T_final=TIT)
    cycle.add_equipment(type='expansion', label='Low Pressure Turbine', P_final=Pmin, efficiency=0.9)
    cycle.add_equipment(type='cooling', label='Main Cooler', T_final=CIT)
    cycle.add_equipment(type='recuperation',
                        label='Recuperator',
                        coolerName='Main Cooler',
                        heaterName='Main Heater',
                        T_warm_margin=5)
    
    return {'cycle': cycle, 'efficiency': cycle._calculate_efficiency(), 'params': kwargs}

if __name__ == "__main__":

    # determining number of subplots required for figure
    subplots = 0
    for key, value in cycles_to_test.items():
        if value == True: subplots += 1
    if subplots == 1:
        rows = 1
        columns = 1
    elif subplots == 2:
        rows = 1
        columns = 2
    elif subplots == 3:
        rows = 1
        columns = 3
    elif subplots == 4:
        rows = 2
        columns = 2
    elif subplots >= 5:
        rows = np.ciel(subplots/3)
        columns = 3
    plt.subplots(rows, columns, sharey=True)
    figure_info = {
        'figure': None,
        'rcount': rows,
        'ccount': columns,
        'subplt': 1, 
        'curcol': 1
    }

    execution = {
        "simple cycle" : lambda: simple_cycle(CIT=CIT, TIT=TIT, Pmax=Pmax, PR=PR),
        "reheat cycle" : lambda: simple_cycle_w_reheat(CIT=CIT, TIT=TIT, Pmax=Pmax, PR=PR),
        "recoup cycle" : lambda: simple_cycle_w_recoup(CIT=CIT, TIT=TIT, Pmax=Pmax, PR=PR),
        "recoup reheat": lambda: recoup_cycle_w_reheat(CIT=CIT, TIT=TIT, Pmax=Pmax, Pmid=Pmid, PR=PR),
    }

    # executing requested instances of the cycle class 
    for key, value in cycles_to_test.items():
        if value == True:
            print(f'\n {key}:')
            instance = execution[key]()
            instance['cycle'].plot_cycle(figure_info)
            instance['cycle'].cycle_info()

    # df = pd.read_csv('results/recuperating brayton cycle with reheat.csv')
    # print(df)

    # plt.scatter(df['Pmid'], df['efficiency'])
    # plt.show()
    # quit()

    plt.show()
    print('\n Model complete. Testing concluded.\n')
