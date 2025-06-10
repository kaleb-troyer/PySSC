# Created on Fri Jun  9 10:56:12 2017
# @author: tneises, ktroyer

#---Core Imports
import pandas as pd
import numpy as np
import os
import csv
from core import sco2_cycle as sco2_solver
from core import sco2_plots as cy_plt

#---Imports for Custom Eval
import matplotlib.pyplot as plt
import addcopyfighandler
import utilities as ut
from pyfluids import Fluid, FluidsList, Input
from collections import defaultdict

# def capacitanceEval(pres=25e6, 
    #                 msand=483.7,
    #                 msalt=387.1,
    #                 msco2=469.0,
    #                 inlet=560.9,
    #                 leave=730.0, 
    #                 sco2_nodes=[],
    #                 sand_nodes=[],
    #                 salt_nodes=[],
    #                 salt_UA=[],
    #                 sand_UA=[]):

    # #---Fluid Properties
    # fluid = Fluid(FluidsList.CarbonDioxide).with_state(
    #     Input.temperature(300), Input.pressure(pres)
    # )

    # #---Sand vs Salt HTF
    # def salt_cp(T_K):
    #     return (1443. + 0.172 * (T_K-273.15))/1000.0
    # def baux_cp(T_K): 
    #     return 0.148 * (T_K**0.3093)
    # def salt_rho(T_K):
    #     return max(2090.0 - 0.636 * (T_K-273.15), 1000) 
    # def baux_rho(T_K):
    #     return 3300 * 0.55
    # def sio2_rho(T_K):
    #     frac = 0.61
    #     T_C = T_K - 273.15

    #     if T_C < 573: 
    #         rho = frac * 2648
    #     elif T_C < 870: 
    #         rho = frac * 2530
    #     elif T_C < 1470: 
    #         rho = frac * 2250
    #     elif T_C < 1705: 
    #         rho = frac * 2200

    #     return rho
    # def sio2_cp(T_K): 
    #     M = 0.0600843
    #     t = T_K / 1000

    #     if T_K < 847: 
    #         A = -6.076591
    #         B =  251.6755
    #         C = -324.7964
    #         D =  168.5604
    #         E =  0.002548
    #     else: 
    #         A =  58.75340
    #         B =  10.27925
    #         C = -0.131384
    #         D =  0.025210
    #         E =  0.025601
        
    #     C1 = A + (B*(t**1)) + (C*(t**2))
    #     C2 = (D*(t**3)) + + (E*(t**(-2)))
    #     cp = (C1 + C2) / (1000 * M)
    #     return cp

    # range_start = 500
    # range_close = 1000

    # temps = []
    # bauxs = []
    # sio2s = []
    # salts = []
    # sco2s = []
    # saltUA = [salt_UA[i]/salt_UA[-1] for i in range(len(salt_UA))]
    # sandUA = [sand_UA[i]/sand_UA[-1] for i in range(len(sand_UA))]

    # for temp in np.arange(range_start, range_close+1, 1 ): 

    #     fluid.update(
    #         Input.temperature(temp), Input.pressure(pres)
    #     )

    #     # baux = baux_cp(temp)
    #     # salt = salt_cp(temp)
    #     # sio2 = sio2_cp(temp)
    #     # sco2 = fluid.specific_heat / 1000
    #     # name = "Specific Heat Capacity"
    #     # dims = "[kJ/kg-K]"
    #     # symb = "cp"

    #     # baux = baux_cp(temp)*sand_rho(temp)
    #     # salt = salt_cp(temp)*salt_rho(Ttemp)
    #     # sco2 = fluid.specific_heat * fluid.density / 1000
    #     # name = "Heat Capacity"
    #     # dims = "[kJ/m3-k]"
    #     # symb = "Cp"

    #     baux = baux_cp(temp) * msand
    #     salt = salt_cp(temp) * msalt
    #     sio2 = sio2_cp(temp) * msand
    #     sco2 = (fluid.specific_heat / 1000) * msco2
    #     name = "Capacitance Rate"
    #     dims = "[kW/k]"
    #     symb = "C"

    #     temps.append(temp)
    #     salts.append(salt)
    #     bauxs.append(baux)
    #     sco2s.append(sco2)
    #     sio2s.append(sio2)

    # plt.subplot(1, 1, 1)
    # plt.axvline(x=inlet, color="black", linestyle="--", linewidth=0.75)
    # plt.axvline(x=leave, color="black", linestyle="--", linewidth=0.75)
    # plt.plot(temps, salts, label="Solar Salt")
    # plt.plot(temps, bauxs, label="Bauxite")
    # plt.plot(temps, sco2s, label="sCO2 (25MPa)")
    # plt.plot(temps, sio2s, label="Silica")
    # plt.legend()
    # plt.xlabel("Temperature [K]")
    # plt.ylabel(name + " " + dims)
    # plt.margins(0)
    # plt.grid(True)
    # plt.title("HTF" + " " + name)

    # # plt.subplot(1, 2, 2)
    # # plt.plot(range(0, len(salt_nodes)), salt_nodes, label="Salt")
    # # plt.plot(range(0, len(sand_nodes)), sand_nodes, label="Sand", linestyle="--")
    # # plt.plot(range(0, len(sco2_nodes)), sco2_nodes, label="sCO2")
    # # plt.legend()
    # # plt.ylabel("Temperature [K]")
    # # plt.xlabel("PHX Node")
    # # plt.margins(0)
    # # plt.grid(True)
    # # plt.title("HTF Subdivision Temperatures")

    # plt.tight_layout()
    # plt.show()

def get_sco2_design_parameters():

    des_par = {}
    des_par["quiet"] = 1                    # [-] If true (1), no status=successful log notices. 
    des_par["opt_logging"] = 0              # [-] If true (1), save each opt loop result to objective.csv.
    des_par["opt_penalty"] = 0              # [-] If true (1), allow addition of penalty terms to objective.

    # --- System design parameters
    des_par["htf"] = 17                     # [-] See design_parameters.txt
    des_par["T_htf_hot_des"] = 670.0        # [C] HTF design hot temperature (PHX inlet)
    des_par["dT_PHX_hot_approach"] = 20.0   # [C/K] default 20. Temperature difference between hot HTF and turbine inlet
    des_par["T_amb_des"] = 35.0             # [C] Ambient temperature at design
    des_par["dT_mc_approach"] = 6.0         # [C] Use 6 here per Neises & Turchi 19. Temperature difference between main compressor CO2 inlet and ambient air
    des_par["site_elevation"] = 588         # [m] Elevation of Daggett, CA. Used to size air cooler...
    des_par["W_dot_net_des"] = 50.0         # [MWe] Design cycle power output (no cooling parasitics)
    des_par["TES_capacity"] = 12.0          # [hours] Thermal engery storage hours

    # --- Cycle design options
        # Configuration
    des_par["cycle_config"] = 1  # [1] = RC, [2] = PC

        # Recuperator design
    des_par["design_method"] = 2        # [-] 1 = specify efficiency, 2 = specify total recup UA, 3 = Specify each recup design (see inputs below)
    des_par["eta_thermal_des"] = 0.44   # [-] Target power cycle thermal efficiency (used when design_method == 1)
    des_par["UA_recup_tot_des"] = 15 * 1000 * (des_par["W_dot_net_des"]) / 50.0  # [kW/K] (used when design_method == 2). If < 0, optimize. 

        # Pressures and recompression fraction
    des_par["is_recomp_ok"] = 1 	# 1 = Yes, 0 = simple cycle only, < 0 = fix f_recomp to abs(input)
    des_par["is_P_high_fixed"] = 1  # 0 = No, optimize. 1 = Yes (=P_high_limit)
    des_par["is_PR_fixed"] = 0      # 0 = No, >0 = fixed pressure ratio at input <0 = fixed LP at abs(input)
    des_par["is_IP_fixed"] = 0      # partial cooling config: 0 = No, >0 = fixed HP-IP pressure ratio at input, <0 = fixed IP at abs(input)
    
    # --- Convergence and optimization criteria
    des_par["des_objective"] = 1 # [2] = hit min phx deltat then max eta, [3] = min cost, [else] max eta
    des_par["rel_tol"] = 3  # [-] Baseline solver and optimization relative tolerance exponent (10^-rel_tol)

    # Weiland & Thimsen 2016
    # In most studies, 85% is an accepted isentropic efficiency for either the main or recompression compressors, and is the recommended assumption.
    des_par["eta_isen_mc"] = 0.85  # [-] Main compressor isentropic efficiency
    des_par["eta_isen_rc"] = 0.85  # [-] Recompressor isentropic efficiency
    des_par["eta_isen_pc"] = 0.85  # [-] Precompressor isentropic efficiency

    # Weiland & Thimsen 2016
    # Recommended turbine efficiencies are 90% for axial turbines above 30 MW, and 85% for radial turbines below 30 MW.
    des_par["eta_isen_t"] = 0.90  # [-] Turbine isentropic efficiency
    des_par["P_high_limit"] = 25  # [MPa] Cycle high pressure limit

    # Weiland & Thimsen 2016
    # Multiple literature sources suggest that recuperator cold side (high pressure) pressure drop of
    # approximately 140 kPa (20 psid) and a hot side (low pressure) pressure drop of 280 kPa (40 psid) can be reasonably used.
    # Note: Unclear what the low pressure assumption is in this study, could be significantly lower for direct combustion cycles
    eff_max = 1.0
    deltaP_recup_HP = 0.0056  # [-] = 0.14[MPa]/25[MPa]
    deltaP_recup_LP = 0.0311  # [-] = 0.28[MPa]/9[MPa]
    
    # --- LTR
    des_par["LTR_design_code"] = 3          # 1 = UA, 2 = min dT, 3 = effectiveness
    des_par["LTR_UA_des_in"] = 2200.0       # [kW/K] (required if LTR_design_code == 1)
    des_par["LTR_min_dT_des_in"] = 12.0     # [C] (required if LTR_design_code == 2)
    des_par["LTR_eff_des_in"] = 0.895       # [-] (required if LTR_design_code == 3)
    des_par["LT_recup_eff_max"] = eff_max   # [-] Maximum effectiveness low temperature recuperator
    des_par["LTR_LP_deltaP_des_in"] = deltaP_recup_LP  # [-]
    des_par["LTR_HP_deltaP_des_in"] = deltaP_recup_HP  # [-]
    
    # --- HTR
    des_par["HTR_design_code"] = 3          # 1 = UA, 2 = min dT, 3 = effectiveness
    des_par["HTR_UA_des_in"] = 2800.0       # [kW/K] (required if LTR_design_code == 1)
    des_par["HTR_min_dT_des_in"] = 19.2     # [C] (required if LTR_design_code == 2)
    des_par["HTR_eff_des_in"] = 0.945       # [-] (required if LTR_design_code == 3)
    des_par["HT_recup_eff_max"] = eff_max   # [-] Maximum effectiveness high temperature recuperator
    des_par["HTR_LP_deltaP_des_in"] = deltaP_recup_LP  # [-]
    des_par["HTR_HP_deltaP_des_in"] = deltaP_recup_HP  # [-]
    
    # --- PHX
    des_par["PHX_co2_deltaP_des_in"] = deltaP_recup_HP  # [-]
    des_par["dT_PHX_cold_approach"]  = 20  # [C/K] default 20. Temperature difference between cold HTF and cold CO2 PHX inlet
    des_par["PHX_n_sub_hx"]   = 10 
    des_par["PHX_cost_model"] = 3

    # --- Air Cooler
    des_par["deltaP_cooler_frac"] = 0.005  # [-] Fraction of CO2 inlet pressure that is design point cooler CO2 pressure drop
    des_par["fan_power_frac"] = 0.02  # [-] Fraction of net cycle power consumed by air cooler fan. 2% here per Turchi et al.
    
    # --- Default
    des_par["deltaP_counterHX_frac"] = 0.0054321  # [-] Fraction of CO2 inlet pressure that is design point counterflow HX (recups & PHX) pressure drop

    return des_par

# Save dictionary of design parameters from above
print("current processID:", os.getpid(), "\n");
design_parameters = get_sco2_design_parameters()
parametric_study = False

#---Cycle Design Simulation (Salt)
# Update Design Parameters
design_parameters["htf"] = 36                   # Silica = 36, Solar Salt = 17
design_parameters["T_htf_hot_des"] = 700        # [C]
design_parameters["des_objective"] = 1          # [1] max eta, [3] minimize cost
design_parameters["W_dot_net_des"] = 100        # [MWe] 
design_parameters["UA_recup_tot_des"] = 30e3    # [kW/K]
design_parameters["PHX_cost_model"] = 100       # [-] 
c_sco2 = sco2_solver.C_sco2_sim(1)  # Initialize to the recompression cycle default (1)
if not parametric_study: 
    # executing the cycle
    c_sco2.overwrite_default_design_parameters(design_parameters)
    c_sco2.solve_sco2_case()            # Run design simulation
    c_sco2.m_also_save_csv = False
    # c_sco2.save_m_solve_dict("design_solution__default_pars")   # Save design solution dictionary
    design_solution_dict_eta = c_sco2.m_solve_dict

    design_parameters["opt_logging"] = 0
    design_parameters["opt_penalty"] = 1

    #---Cycle Design Simulation (Sand) 
    # setting design parameters 
    design_parameters["des_objective"] = 3              # [1] max eta, [3] minimize cost
    design_parameters["W_dot_net_des"] = 100            # [MWe] 
    design_parameters["htf"] = 36                       # Silica = 36, Solar Salt = 17
    # recuperators 
    design_parameters["UA_recup_tot_des"] = -100e3      # [kW/K]
    design_parameters["LTR_min_dT_des_in"] = 5          # [C] 
    design_parameters["HTR_min_dT_des_in"] = 5          # [C] 
    # primary heat exchanger 
    design_parameters["T_htf_hot_des"] = 700            # [C]
    design_parameters["dT_PHX_hot_approach"]  = 20.0    # [C]
    design_parameters["dT_PHX_cold_approach"] = 20.0    # [C]
    design_parameters["PHX_cost_model"] = 100           # [-]
    # executing the cycle 
    c_sco2 = sco2_solver.C_sco2_sim(1)  # Initialize to the recompression cycle default (1)
    c_sco2.overwrite_default_design_parameters(design_parameters)
    c_sco2.solve_sco2_case()            # Run design simulation
    c_sco2.m_also_save_csv = True
    c_sco2.save_m_solve_dict("design_solution__default_pars")   # Save design solution dictionary
    design_solution_dict = c_sco2.m_solve_dict

    #---Plotting a cycle design
    c_plot = cy_plt.C_sco2_TS_PH_plot(design_solution_dict)
    c_plot.is_save_plot = True
    c_plot.file_name = "cycle_design_plots__default_pars"
    c_plot.plot_new_figure()

    dicts = [design_solution_dict_eta, design_solution_dict]
    keys = list(dicts[0].keys())

    csv_file = "comparison.csv"
    with open(csv_file, 'w', newline='') as file: 
        writer = csv.writer(file)
        header = ['parameter'] + [f'{i + 1}' for i in range(len(dicts))]
        writer.writerow(header)

        for key in keys: 
            row = [key] + [d.get(key, '') for d in dicts]
            writer.writerow(row)

#---Parametric Study, 1D
parametric_results = {}
if parametric_study: 
    
    print("Initializing parametric study")
    print("---------------------------------------------")

    parameters = {
        "opt_penalty"   : [1], 
        "T_htf_hot_des" : [700], 
        "des_objective" : [3], 
        "W_dot_net_des" : [100], 
        "PHX_cost_model": [10, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200], 
        "UA_recup_tot_des": [-100e3]
    }

    # parameters = {
    #     "is_recomp_ok": [1], 
    #     "design_objective": [1], 
    #     "T_htf_hot_des" : [700], 
    #     "des_objective" : [1], 
    #     "W_dot_net_des" : [100], 
    #     "PHX_cost_model": [3], 
    #     "UA_recup_tot_des": list(np.arange(5, 51, 1) * 1e3)
    # }

    c_sco2.overwrite_default_design_parameters(design_parameters)

    # generates all parametric study parameter sets
    def generate_combinations(d, current_combination={}, depth=0, results=None):
        if results is None:
            results = []
        
        keys = list(d.keys())
        if depth == len(d):
            results.append(current_combination)
            return results
        
        current_key = keys[depth]
        for value in d[current_key]:
            next_combination = current_combination.copy()
            next_combination[current_key] = value
            generate_combinations(d, next_combination, depth + 1, results)
    
        return results

    par_dict_list = generate_combinations(parameters)

    c_sco2.solve_sco2_parametric(par_dict_list)
    print("\nDid the parametric analyses solve successfully = ",c_sco2.m_par_solve_success)
    c_sco2.m_also_save_csv = True
    c_sco2.save_m_par_solve_dict("parametric_study")
    sol_dict_parametric = c_sco2.m_par_solve_dict
    df = pd.DataFrame(sol_dict_parametric)
    df.to_csv('parametric.csv')

    print(sol_dict_parametric["T_htf_hot_des"])
    print(sol_dict_parametric["PHX_cost_model"])
    print(sol_dict_parametric["levelized_cost_of_energy"])

    def split_and_plot(delineators, y_values, x_values):
        colors = ut.colorGenerator()
        subsets = defaultdict(list)
        for i, d in enumerate(delineators):
            subsets[d].append(y_values[i])
        
        for i, (key, subset) in enumerate(subsets.items()):
            x_subset = x_values[:len(subset)]
            plt.plot(x_subset, subset, label=f'TIT {key}', color=next(colors))
        
        plt.xlabel(r'$\eta_{th}$ [%]')
        plt.ylabel('Cycle Capital Cost [M$]')
        plt.legend()
        plt.grid()
        plt.margins(x=0)
        plt.show()

    split_and_plot(sol_dict_parametric["T_htf_hot_des"], sol_dict_parametric["cycle_cost"], sol_dict_parametric["eta_thermal_calc"])


