
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

'''
Rearranging the M-R-M parameterization to express
stress-to-creep rupture in terms of temperature &
design lifetime.  
'''

b0, b1, b2, b3, sc, T, tr = sy.symbols('b0 b1 b2 b3 sc T tr', positive=True, real=True)

eq = b0 + (b1/T) + (b2 * sy.log(sc, 10)) + (b3 * sy.log(sc, 10) / T) - sy.log(tr, 10)

solution = sy.solve(eq, sc)
solution = solution[0].simplify()

print(solution, '\n')
sy.pprint(solution)


