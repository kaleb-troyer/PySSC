
import matplotlib as mpl
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'  # 'azel', 'trackball', 'sphere', or 'arcball'

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyfluids import Fluid, FluidsList, Input 
from Pyflow.pyflow import Model, Pyflow
import matplotlib.pyplot as plt
import addcopyfighandler
import numpy as np

#---Given Parameters
Dg = 0.01
dg = 0.01
tc = Dg*2
th = 0.01
tm = tc/4
W = 8

m_dot_sco2 = 1010.1
m_dot_sand = 459.20

def Nch_sco2(W, D=Dg, d=dg): 
    return W / (D + d)

def Nch_sand(W, m_dot_sand, th=th): 

    m_dot_sand_flux = 10 # [kg/s/m-m]
    return m_dot_sand / (m_dot_sand_flux * W * th)

def sco2_velocity(Nch_sand, Nch_sco2, m_dot_sco2, D=Dg): 
    return m_dot_sco2 / (Nch_sand * Nch_sco2 * (np.pi / 4) * D**2)

def visualize(H, W, L, Hnom=0.50, Wnom=0.35, Lnom=0.20): 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(int(np.ceil(L / Lnom))): 
        for j in range(int(np.ceil(W / Wnom))): 
            for k in range(int(np.ceil(H / Hnom))): 

                h = Hnom
                w = Wnom 
                l = Lnom

                X = k * Hnom
                Y = j * Wnom
                Z = i * Lnom
                # Define the corners of the box using H, W, and L
                points = np.array([
                    [X+0, Y+0, Z+0],
                    [X+h, Y+0, Z+0],
                    [X+h, Y+w, Z+0],
                    [X+0, Y+w, Z+0],
                    [X+0, Y+0, Z+l],
                    [X+h, Y+0, Z+l],
                    [X+h, Y+w, Z+l],
                    [X+0, Y+w, Z+l]
                ])

                # Define the six faces of the box using the points above
                faces = [
                    [points[j] for j in [0, 1, 2, 3]],  # Bottom face
                    [points[j] for j in [4, 5, 6, 7]],  # Top face
                    [points[j] for j in [0, 1, 5, 4]],  # Front face
                    [points[j] for j in [2, 3, 7, 6]],  # Back face
                    [points[j] for j in [0, 3, 7, 4]],  # Left face
                    [points[j] for j in [1, 2, 6, 5]]   # Right face
                ]

                # Create a Poly3DCollection for the box faces and add to the axis
                ax.add_collection3d(Poly3DCollection(faces, color='red', alpha=0.5, linewidths=1, edgecolor='black'))

                # Set the limits of the axes
                ax.set_xlim([0, max([H, W, L])])
                ax.set_ylim([0, max([H, W, L])])
                ax.set_zlim([0, max([H, W, L])])

    # Show the plot
    plt.show()

if __name__=='__main__': 

    nch_sand = Nch_sand(W, m_dot_sand)
    nch_sco2 = Nch_sco2(W)
    v = sco2_velocity(nch_sand, nch_sco2, m_dot_sco2)

    model = Pyflow(
        Model.Pipe(D=Dg, v=v), 
        Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(600), Input.pressure(25e6)
        )
    )

    km = 30
    hh = 200
    hc = model.htc

    H = nch_sand * (tc + th) + tc
    UA = [92098.15396134065, 102225.98598607304, 114976.98321662343, 131525.35497619555, 153871.32477394858, 185725.7544522962, 234850.0422032107, 320741.4498411041, 511311.05794898287, 1118067.5983674587]
    xt = []
    for UAi in UA: 
        xi = (UAi / (2 * nch_sand * W)) * ((1/hc) + (tm/km) + (1/hh))
        xt.append(xi)

    L = sum(xt)

    structure = 1.2
    vol = L * (structure * W * tc - (np.pi/4) * Dg**2 * nch_sco2) * nch_sand

    print(f"Nch = {nch_sand:.2f}, m_dot_calc {10*nch_sand*W*th:.2f}")
    print(f"Dch = {nch_sco2:.2f}, m_dot_calc {v*(np.pi/4)*(Dg**2)*nch_sand*nch_sco2:.2f}")
    print(f"LxWxH: {L:.3f} X {W:.3f} X {H:.3f}")
    print(f"{vol:.2f} [m3]")

    Lnom = 0.20     # [m]
    Hnom = Dg+dg    # [m]
    Wnom = 0.35     # [m]

    cellcount = 0
    for i in range(int(np.ceil(L / Lnom))): 
        for j in range(int(np.ceil(W / Wnom))): 
            for k in range(int(np.ceil(H / Hnom))): 
                cellcount += 1

    print(cellcount)
    for x in xt: 
        print(f"{x:.3f}")


