
import numpy as np
import pandas as pd
import addcopyfighandler
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example sigmoid function (logistic model)
def sigmoid(x, L, k, x0, c):
    return L / (1 + np.exp(-k * (x - x0))) + c
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))
def linear(x, m, b): 
    return x * m + b
def exponential(x, a, b, c, d): 
    return a * np.exp(b * (x + c)) + d

df = pd.read_csv('mass_flow_rates.csv')
print(df)

x_data = df['Valve']
y_data1 = df['Flat Plates']
y_data2 = df['TO nonu']
y_data3 = df['TO uniform']

params1, _ = curve_fit(sigmoid, x_data, y_data1, p0=[max(y_data1), 1, np.median(x_data), min(y_data1)])
params2, _ = curve_fit(logistic, x_data, y_data2, p0=[max(y_data2), 1, np.median(x_data)])
params3, _ = curve_fit(sigmoid, x_data, y_data3, p0=[max(y_data1), 1, np.median(x_data), min(y_data1)])

# Plot the data and fits
x_fit = np.linspace(min(x_data), max(x_data), 500)
plt.scatter(x_data, y_data1, label='Flat Plates Data', color='blue')
plt.plot(x_fit, sigmoid(x_fit, *params1), color='cyan', label='Flat Plates')

plt.scatter(x_data, y_data2, label='TO Data', color='red')
plt.plot(x_fit, logistic(x_fit, *params2), color='orange', label='TO Nonuniform')

plt.scatter(x_data, y_data3, label='Flat Plates Data', color='green')
plt.plot(x_fit, sigmoid(x_fit, *params3), color='green', label='TO Uniform')

plt.xlabel('Valve')
plt.ylabel('Response')
plt.legend()
plt.show()

print('FP:', params1)
print('TO:', params2)
print('TO:', params3)


