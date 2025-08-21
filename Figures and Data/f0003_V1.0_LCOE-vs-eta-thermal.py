
import os
import numpy as np
import utilities as ut
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from labellines import labelLines
from collections import defaultdict
from datetime import datetime

version = '1.0'
figID   = 'f0003'
savefig = False
display = False

def split_and_plot(delineators, y_values, x_values, savefig=True, display=True):
    date = datetime.now()
    path = os.path.join(os.getcwd(), "Figures and Data")

    plt.legend()
    for line in plt.gca().get_lines():
        line.set_label(None)

    colors = ut.colorGenerator(factor=0.06)
    subsets = defaultdict(list)
    for i, d in enumerate(delineators):
        subsets[d].append(y_values[i])
    
    for i, (key, subset) in enumerate(subsets.items()):
        x_subset = x_values[:len(subset)]
        plt.plot(x_subset, subset, color=next(colors), label=f"TIT {key:.0f}")

    plt.xlabel(r'$\eta_{th}$ [%]')
    plt.ylabel('Levelized Cost of Energy [$/MW-h]')
    plt.legend()
    plt.grid()
    plt.margins(x=0)

    if savefig: plt.savefig(os.path.join(path, "figures", f"{figID}_V{version}_LCOE-vs-eta-thermal.png"), dpi=300, bbox_inches='tight')
    if display: plt.show()

eta_t = [
    0.3818672255863278, 0.39509770978781283, 0.40622333776623215, 0.4153021751681872, 0.42239232029214296, 0.4278694120717378, 0.4320105130043352, 0.4351633203077192, 0.43748968427659324, 0.43921980145520506, 0.4426317589933752, 0.4465174887263438, 0.45017337035039773, 0.4537528071140103, 0.45651113428349993, 0.4589326112611534, 0.4619076465917481, 0.4641846327265273, 0.46580977987518823, 0.4677518413711143, 0.4695390778935876, 0.47120233926910776, 0.4727189046431412, 0.4740633720310677, 0.47549749710047934, 0.47671030933268854, 0.4778162121545482, 0.478964673085156, 0.4799561318661219, 0.4809006160750894, 0.4818169498080041, 0.48264449767762985, 0.4833965929891854, 0.4840834761577209, 0.48482535032308877, 0.485421075188518, 0.48611615379284134, 0.4867068822489987, 0.4872514598743909, 0.48779977745272396, 0.4882480122665164, 0.4887262826430585, 0.4891057153294636, 0.4895984954721629, 0.4899402815818352, 0.490346361737262
]
LCOE = [
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.06515413948932451, 0.06509157138311952, 0.06535673697249525, 0.06483167074781873, 0.06512719865569401, 0.06490828393860701, 0.06481127360900601, 0.06472464477390114, 0.06483743579645451, 0.06482810774175907, 0.06479134192071831, 0.06483829570057935, 0.06476446531184638, 0.06471336240266483, 0.06484662851941445, 0.06481190588146216, 0.06482348074860361, 0.06492689785191957, 0.06486769008985234, 0.06495037126610953, 0.06501317421873684, 0.06499963272592171, 0.06503347973850958, 0.0650023011702413, 0.06510340108770606, 0.06528541655477627, 0.06523588194257418, 0.06522292163701716, 0.06529157307897797, 0.06532628344417689, 0.06540346338230064, 0.06536277730576134, 0.06549521713866362, 0.06545636135545238, 0.06553854562716883, 0.0655399465357094
]
TIT = [700 for _ in range(0, len(LCOE))]

cost_reduct = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
cost_summary = []
for _ in range(0, 21): 
    cost_summary += cost_reduct

split_and_plot(TIT, [x * 1000 for x in LCOE], [x * 100 for x in eta_t], savefig=savefig, display=display)


