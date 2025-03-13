import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Functions
def sigmoid(x, L, k, x0, c):
    return L / (1 + np.exp(-k * (x - x0))) + c

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def linear(x, m, b): 
    return x * m + b

def exponential(x, a, b, c, d): 
    return a * np.exp(b * (x + c)) + d

# Load data
df1 = pd.read_csv('fp_htc.csv')
df2 = pd.read_csv('to_htc.csv')

x_data1 = df1['htc_rat']
y_data1 = df1['mdot_rat']
x_data2 = df2['htc_rat']
y_data2 = df2['mdot_rat']

# Fit Flat Plates with Linear Model
params1, _ = curve_fit(linear, x_data1, y_data1, p0=[0.5, 1])

# Fit TO with Exponential Model (Adjusted Initial Guess and Bounds)
params2, _ = curve_fit(exponential, x_data2, y_data2, 
                       p0=[1, 0.1, 0, min(y_data2)], 
                       bounds=([0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))

# Plot the data and fits
x_fit1 = np.linspace(min(x_data1), max(x_data1), 500)
x_fit2 = np.linspace(min(x_data2), max(x_data2), 500)

plt.scatter(x_data1, y_data1, label='Flat Plates Data', color='blue')
plt.plot(x_fit1, linear(x_fit1, *params1), color='cyan', label='Flat Plates Fit')

plt.scatter(x_data2, y_data2, label='TO Data', color='red')
plt.plot(x_fit2, exponential(x_fit2, *params2), color='orange', label='TO Fit')

plt.xlabel('HTC Ratio')
plt.ylabel('Mdot Ratio')
plt.legend()
plt.show()

print('Flat Plates Fit Parameters:', params1)
print('TO Fit Parameters:', params2)


