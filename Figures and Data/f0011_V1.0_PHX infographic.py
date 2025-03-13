
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sympy as sy
import numpy as np

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


TIT = 700
PIT = 940

Twarm = (655.14, 940.0)
Tcold = (535.14, 700.0)
posit = (0, 1)

plt.plot(posit, Twarm, color='red', label='Sand')
plt.plot(posit, Tcold, color='blue', label=r'sCO$_{2}$')
plt.scatter(posit, Twarm, color='red', zorder=3)
plt.scatter(posit, Tcold, color='blue', zorder=3)
plt.plot([posit[0], posit[0]], [Tcold[0], Twarm[0]], color='gray', linestyle='--') # Cold Approach Temperature
plt.plot([posit[1], posit[1]], [Tcold[1], Twarm[1]], color='gray', linestyle='--') # Hot Approach Temperature

# Add arrows indicating direction of flow
mid_x = (posit[0] + posit[1]) / 2  # Midpoint in x-direction
mid_y_warm = (Twarm[0] + Twarm[1]) / 2  # Midpoint of warm line
mid_y_cold = (Tcold[0] + Tcold[1]) / 2  # Midpoint of cold line

slope_warm = (Twarm[1] - Twarm[0]) / (posit[1] - posit[0]) 
slope_cold = (Tcold[1] - Tcold[0]) / (posit[1] - posit[0]) 
angle_warm = np.degrees(np.arctan(slope_warm)) 
angle_cold = np.degrees(np.arctan(slope_cold)) 

dx = 0.1  # Small step in x-direction
dy_warm = slope_warm * dx
dy_cold = slope_cold * dx

plt.annotate(
    "", xy=(mid_x - dx, mid_y_warm - dy_warm), xytext=(mid_x + dx, mid_y_warm + dy_warm),
    arrowprops=dict(arrowstyle="->", color="red", linewidth=1.0)
)  # Red arrow pointing left along slope

plt.annotate(
    "", xy=(mid_x + dx, mid_y_cold + dy_cold), xytext=(mid_x - dx, mid_y_cold - dy_cold),
    arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.0)
)  # Blue arrow pointing right along slope

# Add labels for TIT and PIT
plt.text(posit[1], TIT, 'Turbine Inlet  \nTemperature (TIT)  ', verticalalignment='bottom', horizontalalignment='right', fontsize=10, color='black')
plt.text(posit[1], PIT, 'PHX Inlet  \nTemperature (PIT)  ', verticalalignment='bottom', horizontalalignment='right', fontsize=10, color='black')

# Add labels for approach temperatures
plt.text(
    posit[0], 25+(Twarm[0] + Tcold[0]) / 2, '  Cold Approach\n  Temperature'+r' $(dT_{c})$', 
    verticalalignment='center', horizontalalignment='left', fontsize=10, color='gray'
)
plt.text(
    posit[1], -20+(Twarm[1] + Tcold[1]) / 2, 'Hot Approach  \nTemperature'+r' $(dT_{h})$  ', 
    verticalalignment='center', horizontalalignment='right', fontsize=10, color='gray'
)
plt.text(
    mid_x-0.07, 5+mid_y_warm, r'$\dot{m}_{sand}$', 
    verticalalignment='center', horizontalalignment='right', fontsize=10, color='red'
)
plt.text(
    mid_x+0.15, mid_y_cold-10, r'$\dot{m}_{sCO_{2}}$', 
    verticalalignment='center', horizontalalignment='right', fontsize=10, color='blue'
)

plt.legend()
plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.xlabel('Position in PHX')
plt.ylabel('Fluid Temperature')
plt.ylim(top=1000)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('PHX profile.png', dpi=300)
plt.show()


