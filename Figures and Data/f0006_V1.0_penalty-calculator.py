import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import os

figID   = 'f0006'
version = '1.0'
savefig = False
display = True

#---assumed parameters
C1 = 5.0 
C2 = 3.6 
C3 = 2.8 
E1 = 1.3
dT = 5.0

#---symbolic solver
# symbols
E2, E3, x = sy.symbols('E2 E3 x', positive=True, real=True)

# equations
A = C2**(dT - C3 - x)
dAdx = sy.diff(A, x)
B = E2 - E3 * x**2
dBdx = sy.diff(B, x)

# solution and display
solution = sy.solve((A - B, dAdx - dBdx), (E2, E3))
for var, sol in solution.items(): 
    print(f'{var}  {sol.subs({x: E1}).evalf():.4f}')

#---substitution and plotting
# subbing into equation B
E2sol = solution[E2].subs({x: E1}).evalf()
E3sol = solution[E3].subs({x: E1}).evalf()
B = B.subs({E2: E2sol, E3: E3sol})

# range to plot
A_range = np.linspace(E1, C1 + dT)
B_range = np.linspace(0, E1)
A_funct = sy.lambdify((x), A, modules='numpy')
B_funct = sy.lambdify((x), B, modules='numpy')

# plotting
plt.plot(A_range, A_funct(A_range))
plt.plot(B_range, B_funct(B_range))
plt.plot([dT, dT], [-0.25, int(B_funct(0)) + (B_funct(0) != int(B_funct(0))) + 0.25], color='#B0B0B0', linestyle=':')
plt.title('Recuperator Approach Temperature Penalty')
plt.margins(x=0, y=0)
plt.xlabel(r'$dT_{approach}$')
plt.ylabel(r'Penalty [$/MWh]')
plt.grid()

path = os.path.join(os.getcwd(), "Figures and Data")
if savefig: plt.savefig(os.path.join(path, "figures", f"{figID}_V{version}_recuperator-dT-penalty.png"), dpi=300, bbox_inches='tight')
if display: plt.show()
