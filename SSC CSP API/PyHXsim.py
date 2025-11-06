
from pyfluids import Fluid, FluidsList, Input 
from Pyflow.pyflow import Model, Pyflow
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import scipy.optimize as opt
# import addcopyfighandler
import seaborn as sns
import pyvista as pv
import numpy as np
# import utilities

class Sand(): 

    def __init__(self):
        
        self.temperature   = None       # [C]
        self.density       = None       # [kg/m3]
        self.enthalpy      = None       # [J/kg]
        self.specific_heat = None       # [J/kg-K]
        self.conductivity  = None       # [W/m-K]
        self.molar_mass    = 0.0600843  # [kg/mol]
        self.bulk_conductivity = 0.27   # [W/m-K]
        self.packing_fraction  = 0.61   # [-]

    def __repr__(self):
        return (
            f"{'T':.<5}{self.temperature:.>15.3f} [C]\n"
            f"{'rho':.<5}{self.density:.>15.3f} [kg/m3]\n"
            f"{'h':.<5}{self.enthalpy:.>15.3f} [J/kg]\n"
            f"{'c':.<5}{self.specific_heat:.>15.3f} [J/kg-K]\n"
            f"{'M':.<5}{self.molar_mass:.>15.3f} [kg/kmol]\n"
            f"{'gamma':.<5}{self.packing_fraction:.>15.3f} [-]"
        )

    def update(self, temperature=None, enthalpy=None, bounds=(25, 1700)): 
        if isinstance(temperature, (float, int, np.ndarray)) and enthalpy is None: 
            self.temperature = temperature
            self._density()
            self._enthalpy()
            self._specific_heat()
            self._conductivity()
        elif isinstance(enthalpy, (float, int, np.ndarray)) and temperature is None: 
            self.enthalpy = enthalpy
            self._temperature(T_min=bounds[0], T_max=bounds[1])
            self._density()
            self._specific_heat()
            self._conductivity()
        else: raise ValueError('Must specify either temperature or enthalpy, but not both.')
        return self

    def _density(self):
        '''
        Calculates the packed bed density of SiO2. 

        Accepts T [C]
        Returns p [kg/m3]
        '''
        if isinstance(self.temperature, np.ndarray):
            density = np.empty_like(self.temperature, dtype=float)
            density[:] = np.nan  # Default value
            
            # Apply conditions using vectorized masking
            mask1 = self.temperature < 573
            mask2 = (self.temperature >= 573) & (self.temperature < 870)
            mask3 = (self.temperature >= 870) & (self.temperature < 1470)
            mask4 = (self.temperature >= 1470) & (self.temperature < 1705)

            density[mask1] = self.packing_fraction * 2648
            density[mask2] = self.packing_fraction * 2530
            density[mask3] = self.packing_fraction * 2250
            density[mask4] = self.packing_fraction * 2200
            
            self.density = density

        else: # if not vectorized
            if self.temperature < 573: 
                self.density = self.packing_fraction * 2648
            elif self.temperature < 870: 
                self.density = self.packing_fraction * 2530
            elif self.temperature < 1470: 
                self.density = self.packing_fraction * 2250
            elif self.temperature < 1705: 
                self.density = self.packing_fraction * 2200

        # if self.temperature < 573: 
        #     self.density = self.packing_fraction * 2648
        # elif self.temperature < 870: 
        #     self.density = self.packing_fraction * 2530
        # elif self.temperature < 1470: 
        #     self.density = self.packing_fraction * 2250
        # elif self.temperature < 1705: 
        #     self.density = self.packing_fraction * 2200

    def _specific_heat(self):
        """
        Calculates specific heat capacity (cp) for a given temperature T (in Kelvin).
        Uses different sets of coefficients depending on the temperature range.
        
        Accepts T [C]
        Returns c [J/kg-K]
        """
        # Temperature units correction
        self._temperature_K = self.temperature + 273.15
        
        # coefficients for different temperature ranges
        coeffs_lower = {'A': -6.076591, 'B': 251.6755, 'C': -324.7964, 'D': 168.5604, 'E': 0.002548}
        coeffs_upper = {'A':  58.75340, 'B': 10.27925, 'C': -0.131384, 'D': 0.025210, 'E': 0.025601}

        # Check if input is an array or scalar
        if isinstance(self._temperature_K, np.ndarray):
            # Initialize an empty array for specific heat
            self.specific_heat = np.empty_like(self._temperature_K, dtype=float)
            
            # Create masks for temperature ranges
            mask_lower = (298 <= self._temperature_K) & (self._temperature_K < 847)
            mask_upper = (847 <= self._temperature_K) & (self._temperature_K <= 1996)
            mask_invalid = ~ (mask_lower | mask_upper)
            
            if np.any(mask_invalid):
                raise ValueError("Temperature out of valid range (298 - 1996 K)")
            
            # Calculate specific heat for the lower range
            t_lower = self._temperature_K[mask_lower] / 1000
            self.specific_heat[mask_lower] = (1 / self.molar_mass) * (
                coeffs_lower['A'] +
                coeffs_lower['B'] * t_lower +
                coeffs_lower['C'] * t_lower**2 +
                coeffs_lower['D'] * t_lower**3 +
                coeffs_lower['E'] / t_lower**2
            )
            
            # Calculate specific heat for the upper range
            t_upper = self._temperature_K[mask_upper] / 1000
            self.specific_heat[mask_upper] = (1 / self.molar_mass) * (
                coeffs_upper['A'] +
                coeffs_upper['B'] * t_upper +
                coeffs_upper['C'] * t_upper**2 +
                coeffs_upper['D'] * t_upper**3 +
                coeffs_upper['E'] / t_upper**2
            )
        else: # if not vectorized
            if 298 <= self._temperature_K < 847:
                coeffs = coeffs_lower
            elif 847 <= self._temperature_K <= 1996:
                coeffs = coeffs_upper
            else: raise ValueError("Temperature out of valid range (298 - 1996 K)")
            
            t = self._temperature_K / 1000
            self.specific_heat = (1 / self.molar_mass) * (
                coeffs['A'] +
                coeffs['B'] * t +
                coeffs['C'] * t**2 +
                coeffs['D'] * t**3 +
                coeffs['E'] / t**2
            )

        # # Temperature units correction
        # self._temperature_K = self.temperature + 273.15
        
        # # Define coefficients for different temperature ranges
        # coeffs_lower = {'A': -6.076591, 'B': 251.6755, 'C': -324.7964, 'D': 168.5604, 'E': 0.002548}
        # coeffs_upper = {'A':  58.75340, 'B': 10.27925, 'C': -0.131384, 'D': 0.025210, 'E': 0.025601}

        # # Select coefficients based on temperature range
        # if 298 <= self._temperature_K < 847:
        #     coeffs = coeffs_lower
        # elif 847 <= self._temperature_K <= 1996:
        #     coeffs = coeffs_upper
        # else: raise ValueError("Temperature out of valid range (298 - 1996 K)")
        
        # # Compute scaled temperature for specific heat calculation
        # t = self._temperature_K / 1000
        
        # # Calculate specific heat capacity
        # self.specific_heat = (1 / self.molar_mass) * (coeffs['A'] + coeffs['B'] * t + coeffs['C'] * t**2 +
        #             coeffs['D'] * t**3 + coeffs['E'] / t**2)

    def _enthalpy(self):
        """
        Calculates enthalpy relative to 298.15 K for a given temperature T (in Kelvin).
        Uses different sets of coefficients depending on the temperature range.
        
        Accepts T [C]
        Returns h [J/kg]
        """
        self.enthalpy = self._get_enthalpy(self.temperature)

    def _get_enthalpy(self, input) -> float:
        """
        Calculates enthalpy relative to 298.15 K for a given temperature T (in Kelvin).
        Uses different sets of coefficients depending on the temperature range.
        
        Accepts T [C]
        Returns h [J/kg]
        """
        # Temperature units correction
        self._temperature_K = input + 273.15
        
        # Define coefficients for different temperature ranges
        coeffs_lower = {'A': -6.076591, 'B': 251.6755, 'C': -324.7964, 'D': 168.5604, 'E': 0.002548, 'F': -917.6893, 'H': -910.8568}
        coeffs_upper = {'A': 58.75340, 'B': 10.27925, 'C': -0.131384, 'D': 0.025210, 'E': 0.025601, 'F': -929.3292, 'H': -910.8568}

        # Check if input is an array or scalar
        if isinstance(self._temperature_K, np.ndarray):
            # Initialize an empty array for enthalpy
            enthalpy = np.empty_like(self._temperature_K, dtype=float)
            
            # Create masks for temperature ranges
            mask_lower = (298 <= self._temperature_K) & (self._temperature_K < 847)
            mask_upper = (847 <= self._temperature_K) & (self._temperature_K <= 1996)
            mask_invalid = ~ (mask_lower | mask_upper)
            
            # Raise an error if any temperatures are out of the valid range
            if np.any(mask_invalid):
                raise ValueError("Temperature out of valid range (298 - 1996 K)")
            
            # Calculate enthalpy for the lower range
            t_lower = self._temperature_K[mask_lower] / 1000
            enthalpy[mask_lower] = ((1 / self.molar_mass) * (
                coeffs_lower['A'] * t_lower +
                coeffs_lower['B'] * t_lower**2 / 2 +
                coeffs_lower['C'] * t_lower**3 / 3 +
                coeffs_lower['D'] * t_lower**4 / 4 +
                coeffs_lower['E'] / t_lower +
                coeffs_lower['F'] - 
                coeffs_lower['H']
            )) * 1000
            
            # Calculate enthalpy for the upper range
            t_upper = self._temperature_K[mask_upper] / 1000
            enthalpy[mask_upper] = ((1 / self.molar_mass) * (
                coeffs_upper['A'] * t_upper +
                coeffs_upper['B'] * t_upper**2 / 2 +
                coeffs_upper['C'] * t_upper**3 / 3 +
                coeffs_upper['D'] * t_upper**4 / 4 +
                coeffs_upper['E'] / t_upper +
                coeffs_upper['F'] - 
                coeffs_upper['H']
            )) * 1000
            
            self.enthalpy = enthalpy
        else: # if not vectorized
            if 298 <= self._temperature_K < 847:
                coeffs = coeffs_lower
            elif 847 <= self._temperature_K <= 1996:
                coeffs = coeffs_upper
            else: raise ValueError(f"Temperature ({self._temperature_K:.2f} K) is out of valid range (298 - 1996 K)")
            
            t = self._temperature_K / 1000
            enthalpy = ((1 / self.molar_mass) * (
                coeffs['A'] * t +
                coeffs['B'] * t**2 / 2 +
                coeffs['C'] * t**3 / 3 +
                coeffs['D'] * t**4 / 4 +
                coeffs['E'] / t +
                coeffs['F'] - 
                coeffs['H']
            )) * 1000

        return enthalpy

        # # Temperature units correction
        # self._temperature_K = input + 273.15
        
        # # Define coefficients for different temperature ranges
        # coeffs_lower = {'A': -6.076591, 'B': 251.6755, 'C': -324.7964, 'D': 168.5604, 'E': 0.002548, 'F': -917.6893, 'H': -910.8568}
        # coeffs_upper = {'A': 58.75340, 'B': 10.27925, 'C': -0.131384, 'D': 0.025210, 'E': 0.025601, 'F': -929.3292, 'H': -910.8568}
        
        # # Select coefficients based on temperature range
        # if 298 <= self._temperature_K < 847:
        #     coeffs = coeffs_lower
        # elif 847 <= self._temperature_K <= 1996:
        #     coeffs = coeffs_upper
        # else: raise ValueError("Temperature out of valid range (298 - 1996 K)")
        
        # # Compute scaled temperature for specific heat calculation
        # t = self._temperature_K / 1000
        
        # # Calculate enthalpy difference
        # enthalpy = ((1 / self.molar_mass) * (coeffs['A'] * t + coeffs['B'] * t**2 / 2 + coeffs['C'] * t**3 / 3 +
        #                 coeffs['D'] * t**4 / 4 + coeffs['E'] / t + coeffs['F'] - coeffs['H'])) * 1000
        # return enthalpy

    def _temperature(self, T_min=25, T_max=1700):
        """
        Solves for temperature given a target enthalpy value using Brent's method.
        
        Accepts h [J/kg]
        Returns T [C]
        """
        
        # Check if enthalpy is an array
        if isinstance(self.enthalpy, np.ndarray):
            # Initialize an array to store temperatures
            temperatures = np.empty_like(self.enthalpy, dtype=float)
            
            # Loop through each target enthalpy value
            for i, h_target in enumerate(self.enthalpy):
                if self._get_enthalpy(T_min) > h_target or self._get_enthalpy(T_max) < h_target:
                    raise ValueError("Target enthalpy is outside the valid range of temperatures.")
                temperatures[i] = opt.brentq(lambda T: self._get_enthalpy(T) - h_target, T_min, T_max)
            self.temperature = temperatures
            
        else: # if not vectorized
            if self._get_enthalpy(T_min) > self.enthalpy or self._get_enthalpy(T_max) < self.enthalpy:
                raise ValueError("Target enthalpy is outside the valid range of temperatures.")
            
            # Solve for temperature using Brent's method
            self.temperature = opt.brentq(lambda T: self._get_enthalpy(T) - self.enthalpy, T_min, T_max)
        return self.temperature

        # # Ensure enthalpy function is well-behaved in the given range
        # if self._get_enthalpy(T_min) > self.enthalpy or self._get_enthalpy(T_max) < self.enthalpy:
        #     raise ValueError("Target enthalpy is outside the valid range of temperatures.")
        
        # # Solve for temperature using brentq root-finding
        # self.temperature = opt.brentq(lambda T: self._get_enthalpy(T) - self.enthalpy, T_min, T_max)

    def _conductivity(self): 
        """
        Solves for SiO2 conductivity, given a temperature.
        """

        self._temperature_K = self.temperature + 273.15
        # getting coefficients based on temperature
        A = np.where(self._temperature_K < 597, 0.00144, 
                    np.where((self._temperature_K >= 597) & (self._temperature_K < 800), 0.00209, 
                            np.where((self._temperature_K >= 800) & (self._temperature_K < 1002), 0.00398, 0.0)))
        
        B = np.where(self._temperature_K < 597, 0.96928, 
                    np.where((self._temperature_K >= 597) & (self._temperature_K < 800), 0.49472, 
                            np.where((self._temperature_K >= 800) & (self._temperature_K < 1002), 0.49472, 2.87)))

        self.conductivity = A * self._temperature_K + B

class PyHX(): 

    def __init__(self, N: int=10, P: float=25e6, duty=210.5e6, warm: tuple=(0, 0), cold: tuple=(0, 0)):
        
        # importing design parameters and flags
        self.silent   = False
        self.duty     = duty
        self.nodes    = N
        self.pressure = P 
        self.T_warm_i = max(warm)
        self.T_warm_o = min(warm)
        self.T_cold_i = min(cold)
        self.T_cold_o = max(cold)

        # initializing defaults
        self._defaults()

    def __repr__(self):
        prel = 10
        posl = 25
        return (
            f"{'eta':.<{prel}}{self.effectiveness:.>{posl}.3f} [-]\n"
            f"{'UAt':.<{prel}}{self.UA/1e6:.>{posl}.3f} [MW/K]\n"
            f"{'mdot':.<{prel}}{self.m_dot_cold:.>{posl}.3f} [kg/s] (cold)\n"
            f"{'mdot':.<{prel}}{self.m_dot_warm:.>{posl}.3f} [kg/s] (warm)\n"
            f"{'dP':.<{prel}}{self.pressure_losses/1000:.>{posl}.3f} [kPa]\n"
            f"{'v(cold)':.<{prel}}{self.pipe._model.velocity:.>{posl}.3f} [m/s]\n"
            f"{'Nch(cold)':.<{prel}}{self.Nchx:.>{posl}.3f} [-]\n"
            f"{'Nch(warm)':.<{prel}}{self.Nchy:.>{posl}.3f} [-]\n"
            f"{'width':.<{prel}}{self.x:.>{posl}.1f} [m]\n"
            f"{'length':.<{prel}}{self.y:.>{posl}.1f} [m]\n"
            f"{'height':.<{prel}}{self.z:.>{posl}.1f} [m]\n"
            f"{'volume':.<{prel}}{self.volume:.>{posl}.3f} [m3]\n"
            f"{'valve':.<{prel}}{self.valve*100:.>{posl}.2f} [%]\n"
            f"{'cell count':.<{prel}}{self.cellcount:.>{posl}} [-]"
        )

    def counterflow(self):
        '''
        All temperatures are expected in Celsius. 
        '''

        # retreiving inlet enthalpies to calculate mass flow rates
        self.sand.update(temperature=self.T_warm_i)
        self.sCO2.update(
            Input.temperature(self.T_cold_i), Input.pressure(self.pressure)
        )

        hhi = self.sand.enthalpy # [J/kg]
        hci = self.sCO2.enthalpy # [J/kg]

        # retreiving outlet enthalpies to calculate mass flow rates
        self.sand.update(temperature=self.T_warm_o)
        self.sCO2.update(
            Input.temperature(self.T_cold_o), Input.pressure(self.pressure)
        )

        hho = self.sand.enthalpy # [J/kg]
        hco = self.sCO2.enthalpy # [J/kg]

        # calculating mass flow rates using an energy balance
        self.m_dot_warm = self.duty / (hhi - hho)
        self.m_dot_cold = self.duty / (hco - hci)

        # enstantiating data structures for PHX subdivisions
        self.nodes_T_warm = [self.T_warm_i] # [C]
        self.nodes_T_cold = [self.T_cold_o] # [C]
        self.nodes_h_warm = [hhi]           # [J/kg]
        self.nodes_h_cold = [hco]           # [J/kg]
        self.nodes_C_warm = []              # [J/kg-K]
        self.nodes_C_cold = []              # [J/kg-K]
        self.nodes_UA     = []              # [W/K]

        # calculating values at each subdivision
        q_dot_i = self.duty / self.nodes
        for i in range(self.nodes): 
            # calculating sand inlet / outlet temperatures
            T_warm_Ni = self.nodes_T_warm[-1]
            h_warm_Ni = self.nodes_h_warm[-1]
            h_warm_No = h_warm_Ni - (q_dot_i / self.m_dot_warm)

            self.sand.update(enthalpy=h_warm_No, bounds=(T_warm_Ni-100, T_warm_Ni))
            T_warm_No = self.sand.temperature
            assert T_warm_No < T_warm_Ni

            # calculating sCO2 inlet / outlet temperatures
            T_cold_No = self.nodes_T_cold[-1]
            h_cold_No = self.nodes_h_cold[-1]
            h_cold_Ni = h_cold_No - (q_dot_i / self.m_dot_cold)
            self.sCO2.update(
                Input.enthalpy(h_cold_Ni), Input.pressure(self.pressure)
            )
            T_cold_Ni = self.sCO2.temperature

            # retrieving specific heat capacities
            T_warm_Navg = (T_warm_Ni + T_warm_No) / 2
            T_cold_Navg = (T_cold_Ni + T_cold_No) / 2

            self.sand.update(temperature=T_warm_Navg)
            self.sCO2.update(
                Input.temperature(T_cold_Navg), Input.pressure(self.pressure)
            )

            c_warm_Navg = self.sand.specific_heat
            c_cold_Navg = self.sCO2.specific_heat

            # calculating capacitance rates, min/max and ratio
            C_dot_warm = c_warm_Navg * self.m_dot_warm
            C_dot_cold = c_cold_Navg * self.m_dot_cold

            C_min = np.min([C_dot_warm, C_dot_cold])
            C_max = np.max([C_dot_warm, C_dot_cold])
            CR = C_min / C_max

            # calculating effectiveness, NTU, and UA
            q_dot_max = C_min * (T_warm_Ni - T_cold_Ni)
            eta_i = q_dot_i / q_dot_max
            if CR < 1: 
                NTU_i = np.log((1 - (eta_i * CR)) / (1 - eta_i)) / (1 - CR)
            else: NTU_i = eta_i / (1 - eta_i)
            UAi = NTU_i * C_min

            # saving results
            self.nodes_T_warm.append(T_warm_No)
            self.nodes_T_cold.append(T_cold_Ni)
            self.nodes_h_warm.append(h_warm_No)
            self.nodes_h_cold.append(h_cold_Ni)
            self.nodes_C_warm.append(C_dot_warm)
            self.nodes_C_cold.append(C_dot_cold)
            self.nodes_UA.append(UAi)

        self.sand.update(temperature=self.T_cold_i)
        self.sCO2.update(
            Input.temperature(self.T_warm_i), Input.pressure(self.pressure)
        )

        q_dot_max = np.min([
            self.m_dot_warm * (np.max(self.nodes_h_warm) - self.sand.enthalpy), 
            self.m_dot_cold * (self.sCO2.enthalpy - np.min(self.nodes_h_cold)), 
        ])

        self.effectiveness = self.duty / q_dot_max
        self.UA = np.sum(self.nodes_UA)
        return self

    def profile(self, bounds=()): 
        plt.plot(np.insert(np.cumsum(self.nodes_UA[::-1])/1e6, 0, 0), self.nodes_T_warm[::-1], color='red', label=r'$T_{sand}$')
        plt.plot(np.insert(np.cumsum(self.nodes_UA[::-1])/1e6, 0, 0), self.nodes_T_cold[::-1], color='blue', label=r'$T_{sCO_{2}}$')
        plt.scatter(np.insert(np.cumsum(self.nodes_UA[::-1])/1e6, 0, 0), self.nodes_T_warm[::-1], color='red', zorder=3)
        plt.scatter(np.insert(np.cumsum(self.nodes_UA[::-1])/1e6, 0, 0), self.nodes_T_cold[::-1], color='blue', zorder=3)

        if bounds: 
            plt.ylim(bounds)

        plt.margins(x=0)
        plt.legend()
        plt.xlabel(r'Cumulative Conductance, UA [MW/K]')
        plt.ylabel(r'Temperature, T [C]')
        plt.grid()

        plt.tight_layout()
        plt.show()

    def sizing(self, Di: float=0, di: float=0, th: float=0, tc: float=0, mat: str='SiC', geo: str='TO'): 
        """
        Calculates the X, Y, and Z dimensions of the primary heat exchanger
        according to requisite mass flow rates and minimizes the total
        volume of material required using an optimizer. 

        ### Attributes

        ---
        **Di**: *float, units=meters*    
        -> Inner diameter of circular channels running through the matrix.  
        **di**: *float, units=meters*    
        -> Minor distance between circular channels in the matrix.   
        **th**: *float, units=meters/second*  
        -> Thickness of an individual hot fluid channel.  
        **tc**: *float, units=nan*   
        -> Thickness of the matrix carrying circular, cold fluid channels.  
        """

        def count_channels_cold(W, D=Di, d=di): 
            return W / (D + d)
        def count_channels_warm(W, valve, th=th, geo=geo): 
            def massflux(valve, geo=geo): 
                if valve > 1 or valve < 0: 
                    raise ValueError(f"Valve position ({valve:.2f}) must be non-negative and less than one.")
                if geo=='FP': 
                    L = 52.25980865
                    K = 0.137203960
                    X = 33.51893626
                    C = 0.887431990
                elif geo=='TO': 
                    L = 42.53434100
                    K = 0.137150260
                    X = 29.42080488
                    C = 0.518871300
                else: raise ValueError(f"Geometry ({geo}) not recognized. Use 'FP' or 'TO'.")
                massflow_neimic = (C + L / (1 + np.exp(-K * (valve*100 - X))))/1000 # [kg/s]
                flowarea_neimic = 0.0003 # [m2]
                self.massflux = massflow_neimic / flowarea_neimic # [kg/s-m2]
                return self.massflux
            
            return self.m_dot_warm / (W * th * massflux(valve))
        def roughness(mat=mat, geo=geo): 
            if mat == 'SiC' and geo == 'TO': 
                e = 150e-6 # [m] wall roughness, meters
            elif mat == 'SiC' and geo == 'FP': 
                e = 5e-6   # [m]
            elif mat == '316H' and geo == 'TO': 
                e = 10e-6  # [m]
            elif mat == '316H' and geo == 'FP': 
                e = 1e-6
            else: raise ValueError(f"Material or geometry not recognized.")
            return e
        def conductivity(T, mat=mat): 
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
        def htc_channel_warm(W, air, dxi, th=th, geo=geo): 

            def htc_channel_base(): 
                #---Given Parameters
                XX = 0.5
                dp = 0.00035
                eps_bk = 0.45

                #---Derived Parameters
                eps_nw = 1 - 0.7293 * (1 - eps_bk)
                ve = self.massflux / self.sand.density
                al = self.sand.bulk_conductivity / (self.sand.density * self.sand.specific_heat)
                Dh = 2 * W * th / (W + th)
                Pe = ve * Dh / al 
                Gz = Dh * Pe / dxi
                Nu = (1/2) * ((2 * 0.866 / (np.sqrt(1/Gz)))**(12/5) + 12**(12/5))**(5/12)
                KK = self.sand.conductivity / air.conductivity
                phi = (1/4) * (((KK - 1) / KK)**2 / (np.log(KK) - ((KK - 1) / KK))) - (1 / (3 * KK))
                knw = air.conductivity * (eps_nw * ((1 - eps_nw) / (2 * phi + 2 * (air.conductivity / (3 * self.sand.conductivity)))))

                # estimated near-near wall contact resistance
                Rc = (dp * XX) / knw
                return (Rc + (2 * th / (Nu * self.sand.bulk_conductivity)))**(-1)
            def htc_ratio(geo=geo): 
                massflux_plate = 7.466 # [kg/s-m2]
                massflux_ratio = self.massflux / massflux_plate
                if geo=='FP': 
                    m =  61.51
                    b = -60.29
                    return (massflux_ratio - b) / m
                elif geo=='TO': 
                    a =  0.2627
                    b =  2.5630
                    c = -0.7454
                    d =  0.5822
                    return (np.log(massflux_ratio - d) / b) - (np.log(a) / b) - c
                else: raise ValueError(f"Geometry ({geo}) not recognized. Use 'FP' or 'TO'.")

            htc_base = htc_channel_base()
            correction = htc_ratio(geo='TO')
            if geo == 'TO': 
                correction = 1.15

            return htc_base * correction
        def routine(W, valve=0.2, dxi=0.2): 

            rough = roughness()
            air = Fluid(FluidsList.Air).with_state(
                Input.temperature(self.T_warm_i), Input.pressure(self.atmosphere)
            )

            self.Nchx = count_channels_cold(W)
            self.Nchy = count_channels_warm(W, valve)

            self.nodes_length = []
            self.nodes_htc_warm = []
            self.nodes_pressure = [self.pressure]

            for i, UA in enumerate(self.nodes_UA):
                UAi = self.nodes_UA[i]
                T_warm_avg = (self.nodes_T_warm[-(i+1)] + self.nodes_T_warm[-(i+2)]) / 2
                T_cold_avg = (self.nodes_T_cold[-(i+1)] + self.nodes_T_cold[-(i+2)]) / 2
                air.update(
                    Input.temperature(T_warm_avg), Input.pressure(self.atmosphere)
                )

                self.sand.update(temperature=T_warm_avg)
                self.sCO2.update(
                    Input.temperature(T_cold_avg), Input.pressure(self.nodes_pressure[-1])
                )

                self.pipe.update(
                    Model.Pipe(D=Di, L=dxi, massflow=self.m_dot_cold/(self.Nchx * (self.Nchy + 1)), e=rough)
                )

                self.nodes_htc_warm.append(htc_channel_warm(W, air, dxi))
                R_cold_i = 1 / (self.Nchx * self.Nchy * np.pi * Di * self.pipe.htc)
                R_warm_i = 1 / (2 * self.Nchy * W * self.nodes_htc_warm[-1])
                R_cond_i = ((tc - Di) / 2) / (2 * self.Nchy * W * conductivity(T=(T_warm_avg + T_cold_avg)/2))

                dxi = UAi * (R_warm_i + R_cond_i + R_cold_i)

                self.nodes_length.append(dxi)
                self.nodes_pressure.append(self.nodes_pressure[-1] - self.pipe.dp)

            self.x = W
            self.y = self.Nchy * (tc + th) + tc
            self.z = sum(self.nodes_length)
            
            if geo=='TO':
                to_factor = 1.1
            else: to_factor = 1.0
            self.volume = self.z * self.Nchy * (to_factor * self.x * tc - (np.pi/4) * Di**2 * self.Nchx)
            self.pressure_losses = self.pressure - self.nodes_pressure[-1]
            self.valve = valve

            return (self.x - self.y)**2 + (1/self.z)
        def unit_cell_division(): 
            self._xgap = 0.05     # [m] gap between cell walls in x dir for fittings
            self._ygap = 0.05     # [m] gap between cell walls in y dir for fittings
            self._zgap = 0.01     # [m] gap between cell walls in z dir for fittings
            self._xnom = 0.25     # [m] max length of unit cell (~10in)
            self._ynom = 0.25     # [m] max thickness of unit cell (~10in)
            self._znom = 0.25     # [m] max height of unit cell (~10in)
            
            self.cellcountx = np.ceil(self.x / self._xnom)
            self.cellcounty = np.ceil(self.y / self._ynom)
            self.cellcountz = np.ceil(self.z / self._znom)

            self._cellx = []
            self._celly = []
            self._cellz = []
            self.cellcount = 0
            for i in range(int(self.cellcountx)): 
                for j in range(int(self.cellcounty)): 
                    for k in range(int(self.cellcountz)): 
                        self._cellx.append((self._xnom / 2) + i * (self._xnom + self._xgap))
                        self._celly.append((self._ynom / 2) + j * (self._ynom + self._ygap))
                        self._cellz.append((self._znom / 2) + k * (self._znom + self._zgap))
                        self.cellcount += 1

            self.x = max(self._cellx) - min(self._cellx) + self._xnom
            self.y = max(self._celly) - min(self._celly) + self._ynom
            self.z = max(self._cellz) - min(self._cellz) + self._znom

        if mat not in {'SiC', '316H'}:
            raise ValueError(f"Material '{mat}' not recognized.")
        else: self._mat = mat
        if geo not in {'TO', 'FP'}:
            raise ValueError(f"Geometry '{geo}' not recognized.")
        else: self._geo = geo

        self.pipe = Pyflow(
            Model.Pipe(D=Di, massflow=self.m_dot_cold*1e-5, e=roughness()), 
            fluid=self.sCO2
        )

        self.pipe.silent = self.silent

        def objective(x):
            W, valve = x
            return routine(W, valve)

        initial_guess = (15, 0.2) # width and valve position
        result = opt.minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=[(5, 30), (0.1, 0.9)]
        )

        routine(result.x[0], result.x[1])
        unit_cell_division()

        self.sCO2.update(
            Input.temperature(np.mean(self.nodes_T_cold)), Input.pressure(np.mean(self.nodes_pressure))
        )

        self.pipe.update(
            Model.Pipe(
                D=Di,
                L=sum(self.nodes_length),
                massflow=self.m_dot_cold/(self.Nchx * (self.Nchy + 1)),
                e=roughness()
            )
        )

    def display(self, joinx=False, joiny=True, joinz=False, palette="coolwarm", savefig=False):
        # Setup colormap and color normalization
        norm = mcolors.Normalize(vmin=min(self.nodes_T_cold), vmax=max(self.nodes_T_cold))
        cmap = sns.color_palette(palette, as_cmap=True)
        zmap = np.cumsum([0] + model.nodes_length)
        Tmap = np.flip(np.array(self.nodes_T_cold))

        # Group cells based on non-joined dimensions.
        # For each dimension, if joinX is True then that coordinate is not used in the key.
        groups = defaultdict(lambda: {'x': [], 'y': [], 'z': []})
        for x, y, z in zip(self._cellx, self._celly, self._cellz):
            key = tuple(
                coord for coord, join in zip((x, y, z), (joinx, joiny, joinz)) if not join
            )
            groups[key]['x'].append(x)
            groups[key]['y'].append(y)
            groups[key]['z'].append(z)

        plotter = pv.Plotter()

        # Helper function to compute center and total length for a dimension.
        def get_dim_params(coords, join, nom_val):
            if join:
                dmin, dmax = min(coords), max(coords)
                center = (dmin + dmax) / 2
                length = dmax - dmin + nom_val
            else:
                center = coords[0]
                length = nom_val
            return center, length

        # Process each group and create the corresponding (possibly joined) box.
        for group in groups.values():
            x_center, x_length = get_dim_params(group['x'], joinx, self._xnom)
            y_center, y_length = get_dim_params(group['y'], joiny, self._ynom)
            z_center, z_length = get_dim_params(group['z'], joinz, self._znom)

            box = pv.Cube(
                center=(x_center, y_center, z_center),
                x_length=x_length,
                y_length=y_length,
                z_length=z_length
            )

            # Apply vertex coloring based on the z coordinate.
            vertices = box.points[:, 2]
            vertex_colors = np.array([cmap(norm(np.interp(zv, zmap, Tmap))) for zv in vertices])
            vertex_colors = (vertex_colors * 255).astype(np.uint8)
            box.point_data['colors'] = vertex_colors

            plotter.add_mesh(box, scalars='colors', rgb=True, opacity=1.0, show_edges=False)

        # Add point labels to the plot.
        plotter.add_point_labels(
            [(self.x/2, self.y, -0.5), (self.x, self.y/2-0.5, -0.5), (self.x+0.7, 0.0, self.z/2)], 
            [f"{self.x:.2f} [m]", f"{self.y:.2f} [m]", f"{self.z:.2f} [m]"],
            justification_horizontal='Center',
            justification_vertical='Center',
            show_points=False,
            fill_shape=False,
            shape_opacity=0.0,
            background_opacity=0.0
        )

        if savefig:
            name = f"{self._geo}-{self._mat} PHX"
            plotter.export_html(name+'.html')
        plotter.show()

    def _defaults(self): 
        # fluid property classes and pyflow htc class
        self.pipe = None
        self.sand = Sand().update(temperature=self.T_warm_i)
        self.sCO2 = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(self.T_cold_i), Input.pressure(self.pressure)
        )

        # constants
        self.atmosphere = 101325    # [Pa]  atmospheric pressure

        # instantiating unsolved values and data structures
        self.pressure_losses = None
        self.nodes_htc_warm  = None
        self.nodes_pressure  = None
        self.effectiveness   = None
        self.nodes_T_warm    = None
        self.nodes_T_cold    = None
        self.nodes_h_warm    = None
        self.nodes_h_cold    = None
        self.nodes_C_warm    = None
        self.nodes_C_cold    = None
        self.nodes_length    = None
        self.m_dot_warm      = None
        self.m_dot_cold      = None
        self.cellcountx      = None
        self.cellcounty      = None
        self.cellcountz      = None
        self.cellcount       = None
        self.nodes_UA        = None
        self.massflux        = None
        self.volume          = None
        self._cellx          = None
        self._celly          = None
        self._cellz          = None
        self._xgap           = None
        self._ygap           = None
        self._zgap           = None
        self._xnom           = None
        self._ynom           = None
        self._znom           = None
        self.valve           = None
        self.Nchx            = None
        self.Nchy            = None
        self._mat            = None
        self._geo            = None
        self.UA              = None
        self.x               = None
        self.y               = None
        self.z               = None

if __name__=='__main__':

    # geometry
    Di = 0.003 # [m] Diameter of the cold channels
    di = 0.003 # [m] Distance between cold channels
    th = 0.003 # [m] Warm channel thickness
    tc = 0.006 # [m] Cold channel thickness

    case = 2
    match case: 
        case 1:
            # basis=~100%
            N = 30
            P = 25e6
            duty = 216.17e6 
            warm = (575.08, 1080.0)  # [C]
            cold = (555.08,  780.0)  # [C]
            mat = 'SiC'
            geo = 'TO'
        case 2: 
            # basis=~225%
            N = 30
            P = 25e6
            duty = 201.74e6
            warm = (607.87, 1060.0)  # [C]
            cold = (587.87,  760.0)  # [C]
            mat = 'SiC'
            geo = 'TO'
        case 3: 
            # 316H, Flat plate
            N = 30
            P = 25e6
            duty = 214.4e6  
            warm = (540.0, 700.0)  # [C]
            cold = (520.0, 680.0)  # [C]
            mat = '316H'
            geo = 'FP'

    model = PyHX(
        N=N, P=P, duty=duty, 
        warm=warm, 
        cold=cold  
    ).counterflow()

    model.silent = False
    model.sizing(
        Di = Di, # [m] Diameter of the cold channels
        di = di, # [m] Distance between cold channels
        th = th, # [m] Warm channel thickness
        tc = tc, # [m] Cold channel thickness
        mat = mat, 
        geo = geo, 
    )
    
    model.profile(bounds=(500, 1100))
    model.display(joinx=True, joiny=True, joinz=False, savefig=False)

    print()
    print(model)
    print()
    print(model.pipe)
    print()
    
