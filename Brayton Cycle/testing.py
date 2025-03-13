import numpy as np
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input

Phi = 25e6 # [Pa]
co2 = Fluid(FluidsList.CarbonDioxide).with_state(
    Input.temperature(300), Input.pressure(Phi)
)

temps = np.arange(300, 1300, 1)
specs = []
sand = []

for temp in temps:
    co2.update(Input.temperature(temp), Input.pressure(Phi))
    spec = co2.specific_heat
    specs.append(spec/1000)
    sand.append(0.148 * (temp**0.3093))

plt.plot(temps, sand)
plt.plot(temps, specs)
plt.xlabel("Temperature [K]")
plt.ylabel("Specific Heat Capacity [kJ / kg-K]")
plt.title("Specific Heat of sCO2 at 25MPa")
plt.show()

