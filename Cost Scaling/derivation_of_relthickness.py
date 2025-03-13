
import sympy as sy
import numpy as np

'''
Rearranging the von Mises stress for a thick-walled
cylindrical pressure vessel to create an expression
for thickness in terms of ri, stress, and pressure. 
'''

th, ro, ri, sc, dP = sy.symbols('th ro ri sc dP', positive=True, real=True)

# average vonMises stress in a thick-walled cylinder
E1 = sy.sqrt(3) * (ri * ro / (ro**2 - ri**2)) * dP - sc
E2 = ro - th - ri

f_ri = sy.solve(E1, ri)[0]
f_th = sy.solve(E2, th)[0]
f_th = f_th.subs(ri, f_ri)

f_th = f_th.simplify()
sy.pprint(f_th)

