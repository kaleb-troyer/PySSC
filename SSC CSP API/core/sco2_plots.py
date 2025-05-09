
# Created on Tue Mar 13 13:03:43 2018
# @author: tneises

import matplotlib.pyplot as plt
import math
import copy
import pandas as pd
import numpy as np
import json
import string
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from core import sco2_cycle as py_sco2
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from enum import Enum

def filter_dict_keys(data, keys):
    return {k:v for (k,v) in data.items() if k in keys}

def filter_dict_to_index(data, i):
    data_new = {}
    for key in data.keys():
        data_new[key] = data[key][i]
    return data_new
        
def filter_dict_column_length(data, n_len):
    data_new = {}
    for key in data.keys():
        if(len(data[key]) == n_len):
            data_new[key] = data[key]
        
    return data_new

def filter_dict_index_and_keys(data, i, keys):
    return filter_dict_to_index(filter_dict_keys(data, keys),i)

def ceil_nearest_base(x, base):
    return int(base * math.ceil(float(x)/base))

def floor_nearest_base(x, base):
    return int(base * math.floor(float(x)/base))
  
class C_sco2_cycle_TS_plot:

    def __init__(self, dict_cycle_data):
        self.dict_cycle_data = dict_cycle_data
        self.is_save_plot = False
        self.is_add_recup_in_out_lines = True
        self.is_annotate = True
        self.is_annotate_HTR = True
        self.is_annotate_LTR = True
        self.is_annotate_PHX = True
        self.is_annotate_cooler = True
        self.is_add_P_const_lines = True
        self.is_add_dome = True
        self.is_add_title = True
        self.is_overwrite_title = ""
        self.file_name = ""
        self.lc = 'k'
        self.mt = 'o'
        self.markersize = 4
        self.y_max = -1
        
    def plot_new_figure(self):
        
        fig1, ax1 = plt.subplots(num = 1,figsize=(7.0,4.5))
    
        self.plot_from_existing_axes(ax1)
        
        plt.tight_layout(pad=0.0,h_pad=.30,rect=(0.02,0.01,0.99,0.98))
        
        if(self.is_save_plot):
            
            str_file_name = cycle_label(self.dict_cycle_data, False, True) + "__TS_plot.png"
            
            if(self.file_name != ""):
                str_file_name = self.file_name + ".png"
                                       
            fig1.savefig('results/' + str_file_name)
            
            plt.close()

    def set_y_max(self):

        T_htf_hot = self.dict_cycle_data["T_htf_hot_des"]
        T_t_in = self.dict_cycle_data["T_turb_in"]
        self.y_max = max(self.y_max, max(ceil_nearest_base(T_htf_hot, 100.0), 100 + ceil_nearest_base(T_t_in, 100.0)))

    def plot_from_existing_axes(self, ax_in):
        
        # eta_str = "Thermal Efficiency = " + '{:.1f}'.format(self.dict_cycle_data["eta_thermal_calc"]*100) + "%"

        plot_title = self.is_overwrite_title
    
        # if(self.dict_cycle_data["cycle_config"] == 1):
        #     if(self.is_overwrite_title == ""):
        #         plot_title = "Recompression Cycle: " + eta_str
        # else:
        #     if(self.is_overwrite_title == ""):
        #         plot_title = "Partial Cooling Cycle, " + eta_str
                
        if (self.is_overwrite_title == ""):
            plot_title = get_plot_name(self.dict_cycle_data)

        self.set_y_max()
        
        self.overlay_cycle_data(ax_in)
            
        if(self.is_add_recup_in_out_lines):
            self.add_recup_in_out_lines(ax_in)
            
        if(self.is_annotate):
            self.annotate(ax_in)
            
        self.format_axes(ax_in, plot_title)
        
        return ax_in
    
    def overlay_cycle_data(self, ax_in):
        
        if(self.dict_cycle_data["cycle_config"] == 1):
            self.plot_RC_points_and_lines(ax_in)
        else:
            self.plot_PC_points_and_lines(ax_in)
            
        self.set_y_max()
        
    
    def format_axes(self, ax_in, plot_title):
    
        ax_in.autoscale()
        x_low, x_high = ax_in.get_xlim()
        
        if(self.is_add_P_const_lines):
            self.plot_constP(ax_in)      # add_Ts_constP(ax_in)

        if(self.is_add_dome):
            self.plot_dome(ax_in)       # add_Ts_dome(ax_in)
    
        ax_in.grid(alpha=0.5,which='major')
        ax_in.grid(alpha=0.3,which='minor')
        
        y_down, y_up = ax_in.get_ylim()
        y_min = 0
        ax_in.set_ylim(y_min, self.y_max)
        ax_in.set_xlim(x_low)
        y_down, y_up = ax_in.get_ylim()
        major_y_ticks = np.arange(y_min,y_up+1,100)
        minor_y_ticks = np.arange(y_min,y_up+1,20)
        ax_in.set_yticks(major_y_ticks)
        ax_in.set_yticks(minor_y_ticks,minor=True)
        ax_in.set_ylabel("Temperature [C]", fontsize = 12)
        ax_in.set_xlabel("Entropy [kJ/kg-K]", fontsize = 12)
        
        if(self.is_add_title):
            ax_in.set_title(plot_title, fontsize = 14)
            
        return ax_in
    
    def plot_dome(self, ax_in):
    
        fileDir = os.path.dirname(os.path.abspath(__file__))
        
        dome_data = pd.read_csv(fileDir + "/property_data/ts_dome_data.txt")
        ax_in.plot(dome_data["s"], dome_data["T"], 'k-', lw = 1, alpha = 0.4)
    
    def annotate(self, ax_in):
    
        if(self.is_annotate_HTR and (not(math.isnan(float(self.dict_cycle_data["q_dot_HTR"]))))):
            HTR_title = r'$\bf{High}$' + " " + r'$\bf{Temp}$' + " " + r'$\bf{Recup}$'
            q_dot_text = "\nDuty = " + '{:.1f}'.format(self.dict_cycle_data["q_dot_HTR"]) + " MWt"
            UA_text = "\nUA = " + '{:.1f}'.format(self.dict_cycle_data["HTR_UA_calculated"]) + " MW/K"
            eff_text = "\n" + r'$\epsilon$' + " = " + '{:.3f}'.format(self.dict_cycle_data["eff_HTR"])
            mindt_text = "\n" + r'$\Delta$' + r'$T_{min}$' + " = " + '{:.1f}'.format(self.dict_cycle_data["HTR_min_dT"]) + " C"

            #r'$\eta_{isen}$'

            T_HTR_LP_data = self.dict_cycle_data["T_HTR_LP_data"]
            s_HTR_LP_data = self.dict_cycle_data["s_HTR_LP_data"]
            
            n_p = len(T_HTR_LP_data)
            n_mid = (int)(n_p/2)-2
        
            HTR_text = HTR_title + q_dot_text + UA_text + eff_text + mindt_text
            
            ax_in.annotate(HTR_text, xy=(s_HTR_LP_data[n_mid],T_HTR_LP_data[n_mid]), 
                           xytext=(s_HTR_LP_data[n_mid] + 0.25,T_HTR_LP_data[n_mid]), va="center",
                           arrowprops = dict(arrowstyle="->", color = 'r', ls = '--', lw = 0.6),
                           fontsize = 8,
                           bbox=dict(boxstyle="round", fc="w", pad = 0.5))
        
        if(self.is_annotate_LTR):
            LTR_title = r'$\bf{Low}$' + " " + r'$\bf{Temp}$' + " " + r'$\bf{Recup}$'
            q_dot_text = "\nDuty = " + '{:.1f}'.format(self.dict_cycle_data["q_dot_LTR"]) + " MWt"
            UA_text = "\nUA = " + '{:.1f}'.format(self.dict_cycle_data["LTR_UA_calculated"]) + " MW/K"
            eff_text = "\n" + r'$\epsilon$' + " = " + '{:.3f}'.format(self.dict_cycle_data["eff_LTR"])
            mindt_text = "\n" + r'$\Delta$' + r'$T_{min}$' + " = " + '{:.1f}'.format(self.dict_cycle_data["LTR_min_dT"]) + " C"

            T_LTR_LP_data = self.dict_cycle_data["T_LTR_LP_data"]
            s_LTR_LP_data = self.dict_cycle_data["s_LTR_LP_data"]

            T_HTR_LP_data = self.dict_cycle_data["T_HTR_LP_data"]
            s_HTR_LP_data = self.dict_cycle_data["s_HTR_LP_data"]
            
            n_p = len(T_LTR_LP_data)
            n_mid = (int)(n_p/2) + 5
            
            LTR_text = LTR_title + q_dot_text + UA_text + eff_text + mindt_text
            
            ax_in.annotate(LTR_text, xy=(s_LTR_LP_data[n_mid],T_LTR_LP_data[n_mid]), 
                           xytext=(s_HTR_LP_data[n_mid],T_LTR_LP_data[n_mid]), va="center",
                           arrowprops = dict(arrowstyle="->", color = 'b', ls = '--', lw = 0.6),
                           fontsize = 8,
                           bbox=dict(boxstyle="round", fc="w", pad = 0.5))
        
        if(self.is_annotate_PHX and self.dict_cycle_data["od_T_t_in_mode"] == 0):
            T_states = self.dict_cycle_data["T_state_points"]
            s_states = self.dict_cycle_data["s_state_points"]
            
            dT_PHX_hot_approach = self.dict_cycle_data["dT_PHX_hot_approach"]
            T_PHX_in = T_states[5] + dT_PHX_hot_approach
            s_PHX_in = s_states[5]

            dT_PHX_cold_approach = self.dict_cycle_data["dT_PHX_cold_approach"]
            T_PHX_out = T_states[4] + dT_PHX_cold_approach
            s_PHX_out = s_states[4]

            if self.dict_cycle_data['htf'] == 50: # User-defined properties
                # Plotting Phase Change in HTF
                # get T_vs_cp data
                T = list() 
                cp = list()
                for tdata in self.dict_cycle_data['htf_props']:
                    T.append(tdata[0])
                    cp.append(tdata[1])

                def find_PCM_temps(Temp:list, specific_heat:list, T_HTF_PHX_out:float, T_HTF_PHX_in:float) -> list:
                    PCM_temps = list()
                    cp_prev = 0.0
                    for t, cp in zip(Temp, specific_heat):
                        if abs(cp_prev - cp)/cp >= 0.05:
                            PCM_temps.append(t)
                        cp_prev = cp
                    # end point
                    PCM_temps = [t for t in PCM_temps if t < T_HTF_PHX_in]
                    PCM_temps.append(T_HTF_PHX_in)

                    if PCM_temps[-1] > PCM_temps[0]:
                        PCM_temps.reverse() # reverse order if ascending 

                    PCM_temps = [t for t in PCM_temps if t > T_HTF_PHX_out]
                    PCM_temps.append(T_HTF_PHX_out)
                    return PCM_temps

                PCM_temps = find_PCM_temps(T, cp, T_PHX_out, T_PHX_in)

                s_PHX = [s_PHX_in]
                T_co2 = T_states[5]
                sco2_PHX_avg_Cp = 1.253952381  # [kJ/kg-K] # This is just an approximation 
                for i in range(1, len(PCM_temps)):
                    cp_htf = np.interp((PCM_temps[i-1] + PCM_temps[i])/2, T, cp)
                    q_dot = self.dict_cycle_data['m_dot_htf_des']*cp_htf*(PCM_temps[i-1] - PCM_temps[i])
                    T_co2 = T_co2 - q_dot / (self.dict_cycle_data['m_dot_co2_full'] * sco2_PHX_avg_Cp)
                    s_PHX.append(np.interp(T_co2, self.dict_cycle_data['T_PHX_data'], self.dict_cycle_data['s_PHX_data']))
                s_PHX[-1] = s_PHX_out 

                ax_in.plot(s_PHX, PCM_temps, color = '#ff9900', ls = "-")
                s_PHX_avg = 0.90*s_PHX_in + 0.10*s_PHX[1]
                T_PHX_avg = 0.90*T_PHX_in + 0.10*PCM_temps[1]
            else:
                ax_in.plot([s_PHX_in, s_PHX_out], [T_PHX_in, T_PHX_out], color = '#ff9900', ls = "-")
                s_PHX_avg = 0.90*s_PHX_in + 0.10*s_PHX_out
                T_PHX_avg = 0.90*T_PHX_in + 0.10*T_PHX_out
            
            PHX_title = r'$\bf{Primary}$' + " " + r'$\bf{HX}$'
            q_dot_text = "\nDuty = " + '{:.1f}'.format(self.dict_cycle_data["q_dot_PHX"]) + " MWt"
            htf_text ="\n" + r'$\.m_{htf}$' + " = " + '{:.1f}'.format(self.dict_cycle_data['m_dot_htf_des']) + " kg/s"
            UA_text = "\nUA = " + '{:.1f}'.format(self.dict_cycle_data["UA_PHX"]) + " MW/K"
            eff_text = "\n" + r'$\epsilon$' + " = " + '{:.3f}'.format(self.dict_cycle_data["eff_PHX"])
            mindt_text = "\n" + r'$\Delta$' + r'$T_{min}$' + " = " + '{:.1f}'.format(self.dict_cycle_data["PHX_min_dT"]) + " C"
            
            PHX_text = PHX_title + q_dot_text + htf_text + UA_text + eff_text + mindt_text
            
            ax_in.annotate(PHX_text, xy=(s_PHX_avg, T_PHX_avg), 
                           xytext=(s_PHX_avg-0.25,T_PHX_avg),va="center", ha="right",multialignment="left",
                           arrowprops = dict(arrowstyle="->", color = '#ff9900', ls = '--', lw = 0.6),
                           fontsize = 8,
                           bbox=dict(boxstyle="round", fc="w", pad = 0.5))

        if(self.is_annotate_cooler):

            T_states = self.dict_cycle_data["T_state_points"]
            s_states = self.dict_cycle_data["s_state_points"]

            mc_cool_title = r'$\bf{Main\ Cooler}$'
            T_amb_text = "\n" + r'$T_{amb}\ =\ $' + '{:.1f}'.format(self.dict_cycle_data["T_amb_des"]) + " C"
            T_cold_text = "\n" + r'$T_{out}\ =\ $' + '{:.1f}'.format(self.dict_cycle_data["T_comp_in"]) + " C"
            # if (self.dict_cycle_data["cycle_config"] == 2):
            #     s_q_dot = "IP_cooler_q_dot"
            #     s_W_dot = "IP_cooler_W_dot_fan"
            # else:
            s_q_dot = "mc_cooler_q_dot"
            s_W_dot = "mc_cooler_W_dot_fan"
            q_dot_text = "\nDuty = " + '{:.1f}'.format(self.dict_cycle_data[s_q_dot]) + " MWt"
            W_dot_text = "\n" + r'$\.W_{fan}\ =\ $' + '{:.2f}'.format(self.dict_cycle_data[s_W_dot]) + " MWe"

            T_main_cooler_data = self.dict_cycle_data["T_main_cooler_data"]
            s_main_cooler_data = self.dict_cycle_data["s_main_cooler_data"]

            n_p = len(T_main_cooler_data)
            n_mid = (int)(n_p / 2) + 3

            mc_cool_text = mc_cool_title + T_amb_text + T_cold_text + q_dot_text + W_dot_text

            ax_in.annotate(mc_cool_text, xy=(s_main_cooler_data[n_mid], T_main_cooler_data[n_mid]),
                           xytext=(s_main_cooler_data[n_mid], T_states[3]+50), ha="center",multialignment="left",
                           arrowprops=dict(arrowstyle="->", color='purple', ls='--', lw=0.6),
                           fontsize=8,
                           bbox=dict(boxstyle="round", fc="w", pad=0.5))

            if (self.dict_cycle_data["cycle_config"] == 2):

                pc_cool_title = r'$\bf{Pre\ Cooler}$'
                T_pc_cold_txt = "\n" + r'$T_{out}\ = \ $' + '{:.1f}'.format(self.dict_cycle_data["pc_T_in_des"]) + " C"
                q_dot_pc_txt = "\nDuty = " + '{:.1f}'.format(self.dict_cycle_data["pc_cooler_q_dot"]) + " MWt"
                W_dot_pc_txt = "\n" + r'$\.W_{fan}\ = \ $' + '{:.2f}'.format(self.dict_cycle_data["pc_cooler_W_dot_fan"]) + " MWe"

                T_pc_cooler_data = self.dict_cycle_data["T_pre_cooler_data"]
                s_pc_cooler_data = self.dict_cycle_data["s_pre_cooler_data"]

                n_p = len(T_pc_cooler_data)
                n_q = (int)(0.25*n_p)

                pc_cool_text = pc_cool_title + T_pc_cold_txt + q_dot_pc_txt + W_dot_pc_txt

                ax_in.annotate(pc_cool_text, xy=(s_pc_cooler_data[n_q], T_pc_cooler_data[n_q]),
                               xytext=(s_pc_cooler_data[n_q], T_states[3] + 150), ha="center",
                               multialignment="left",
                               arrowprops=dict(arrowstyle="->", color='purple', ls='--', lw=0.6),
                               fontsize=8,
                               bbox=dict(boxstyle="round", fc="w", pad=0.5))
            
        return ax_in
    
    def add_recup_in_out_lines(self, ax_in):
    
        T_states = self.dict_cycle_data["T_state_points"]
        s_states = self.dict_cycle_data["s_state_points"]
        
        T_LTR_hot = [T_states[2],T_states[7]]
        s_LTR_hot = [s_states[2],s_states[7]]
        ax_in.plot(s_LTR_hot, T_LTR_hot, 'b-.', lw = 0.7, alpha = 0.9)
        
        T_LTR_cold = [T_states[1],T_states[8]]
        s_LTR_cold = [s_states[1],s_states[8]]
        ax_in.plot(s_LTR_cold, T_LTR_cold, 'b-.', lw = 0.7, alpha = 0.9)
        
        T_HTR_cold = [T_states[3],T_states[7]]
        s_HTR_cold = [s_states[3],s_states[7]]
        ax_in.plot(s_HTR_cold, T_HTR_cold, 'r-.', lw = 0.7, alpha = 0.9)
        
        T_HTR_hot = [T_states[4],T_states[6]]
        s_HTR_hot = [s_states[4],s_states[6]]
        ax_in.plot(s_HTR_hot, T_HTR_hot, 'r-.', lw = 0.7, alpha = 0.9)
        
        return ax_in
        
    def plot_hx(self, ax_in):
        
        T_LTR_HP_data = self.dict_cycle_data["T_LTR_HP_data"]
        s_LTR_HP_data = self.dict_cycle_data["s_LTR_HP_data"]
        ax_in.plot(s_LTR_HP_data, T_LTR_HP_data, self.lc)
        
        T_HTR_HP_data = self.dict_cycle_data["T_HTR_HP_data"]
        s_HTR_HP_data = self.dict_cycle_data["s_HTR_HP_data"]
        ax_in.plot(s_HTR_HP_data, T_HTR_HP_data, self.lc)
        
        T_PHX_data = self.dict_cycle_data["T_PHX_data"]
        s_PHX_data = self.dict_cycle_data["s_PHX_data"]
        ax_in.plot(s_PHX_data, T_PHX_data, self.lc)
        
        T_HTR_LP_data = self.dict_cycle_data["T_HTR_LP_data"]
        s_HTR_LP_data = self.dict_cycle_data["s_HTR_LP_data"]
        ax_in.plot(s_HTR_LP_data, T_HTR_LP_data, self.lc)
        
        T_LTR_LP_data = self.dict_cycle_data["T_LTR_LP_data"]
        s_LTR_LP_data = self.dict_cycle_data["s_LTR_LP_data"]
        ax_in.plot(s_LTR_LP_data, T_LTR_LP_data, self.lc)
    
        T_main_cooler_data = self.dict_cycle_data["T_main_cooler_data"]
        s_main_cooler_data = self.dict_cycle_data["s_main_cooler_data"]
        ax_in.plot(s_main_cooler_data, T_main_cooler_data, self.lc)
        
        T_pre_cooler_data = self.dict_cycle_data["T_pre_cooler_data"]
        s_pre_cooler_data = self.dict_cycle_data["s_pre_cooler_data"]
        ax_in.plot(s_pre_cooler_data, T_pre_cooler_data, self.lc)
        
        return ax_in
    
    def plot_RC_points_and_lines(self, ax_in):
        
        self.plot_hx(ax_in)

        T_states = self.dict_cycle_data["T_state_points"]
        s_states = self.dict_cycle_data["s_state_points"]
        
        T_mc_plot = [T_states[0],T_states[1]]
        s_mc_plot = [s_states[0],s_states[1]]
        ax_in.plot(s_mc_plot, T_mc_plot, self.lc)
        
        T_t_plot = [T_states[5],T_states[6]]
        s_t_plot = [s_states[5],s_states[6]]
        ax_in.plot(s_t_plot, T_t_plot, self.lc)
        
        ax_in.plot(s_states[0:10], T_states[0:10], self.lc + self.mt, markersize = self.markersize)
            
        f_recomp = self.dict_cycle_data["recomp_frac"]
            
        if(f_recomp > 0.01):
            T_rc_plot = [T_states[8],T_states[9]]
            s_rc_plot = [s_states[8],s_states[9]]
            ax_in.plot(s_rc_plot, T_rc_plot, self.lc)
        
        return ax_in
    
    def plot_PC_points_and_lines(self, ax_in):
        
        self.plot_hx(ax_in)

        T_states = self.dict_cycle_data["T_state_points"]
        s_states = self.dict_cycle_data["s_state_points"]
        
        T_mc_plot = [T_states[0],T_states[1]]
        s_mc_plot = [s_states[0],s_states[1]]
        ax_in.plot(s_mc_plot, T_mc_plot, self.lc)
        
        T_t_plot = [T_states[5],T_states[6]]
        s_t_plot = [s_states[5],s_states[6]]
        ax_in.plot(s_t_plot, T_t_plot, self.lc)
        
        T_pc_plot = [T_states[10],T_states[11]]
        s_pc_plot = [s_states[10],s_states[11]]
        ax_in.plot(s_pc_plot, T_pc_plot, self.lc)
        
        f_recomp = self.dict_cycle_data["recomp_frac"]
        if(f_recomp > 0.01):
            T_rc_plot = [T_states[11],T_states[9]]
            s_rc_plot = [s_states[11],s_states[9]]
            ax_in.plot(s_rc_plot, T_rc_plot, self.lc)
        
        line, = ax_in.plot(s_states, T_states, self.lc+self.mt, markersize = 4)
        
        return ax_in
    
    def plot_constP(self, ax_in):

        fileDir = os.path.dirname(os.path.abspath(__file__))
        
        P_data = pd.read_csv(fileDir + "/property_data/constantP_data.txt")
        P_vals = []
        for names in P_data.columns.values.tolist():
            if names.split("_")[1] not in P_vals:
                P_vals.append(names.split("_")[1])

        v_n_high = []
        v_n_low = []
        for vals in P_vals:
            for i in range(len(P_data["s_"+P_vals[0]].values)):
                if(P_data["T_"+vals].values[i] > self.y_max-30):
                    v_n_high.append(i-1)
                    v_n_low.append(round((i-1)*0.98))
                    break
                if(i == len(P_data["s_"+P_vals[0]].values) -1):
                    v_n_high.append(i - 1)
                    v_n_low.append(round((i - 1) * 0.98))

        #n_high = len(P_data["s_"+P_vals[0]].values)  - 1
    
        #n_low = round(n_high*0.98)

        T_label = P_data["T_"+P_vals[0]].values[v_n_high[0]]

        label_pressure = True
    
        for i, vals in enumerate(P_vals):
            ax_in.plot(P_data["s_"+vals].values[0:v_n_high[i]], P_data["T_"+vals].values[0:v_n_high[i]], 'k--', lw = 0.5, alpha = 0.4)
            if(label_pressure):
                ax_in.annotate("Pressure (MPa):",[P_data["s_"+vals].values[v_n_low[i]], P_data["T_"+vals].values[v_n_low[i]]], color = 'k', alpha = 0.4, fontsize = 8)
                label_pressure = False
            ax_in.annotate(vals,[P_data["s_"+vals].values[v_n_high[i]], T_label], color = 'k', alpha = 0.4, fontsize = 8)

def get_plot_name(dict_cycle_data):
    
    eta_str = "Thermal Efficiency = " + '{:.1f}'.format(dict_cycle_data["eta_thermal_calc"] * 100) + "%"

    if (dict_cycle_data["cycle_config"] == 1 and dict_cycle_data["is_recomp_ok"] == 1):
        plot_title = "Recompression Cycle, " + eta_str
    elif (dict_cycle_data["cycle_config"] == 1):
        plot_title = "Simple Cycle, " + eta_str
    else:
        plot_title = "Partial Cooling Cycle, " + eta_str
    
    return plot_title

class C_sco2_cycle_PH_plot:
    
    def __init__(self, dict_cycle_data):
        self.dict_cycle_data = dict_cycle_data
        self.is_save_plot = False
        self.is_annotate = True
        self.is_annotate_MC = True
        self.is_annotate_RC = True
        self.is_annotate_PC = True
        self.is_annotate_T = True
        self.is_annotate_MC_stages = False
        self.is_annotate_RC_stages = False
        self.is_add_T_const_lines = True
        self.is_add_dome = True
        self.is_add_title = True
        self.is_overwrite_title = ""
        self.file_name = ""
        self.lc = 'k'
        self.mt = 'o'
        self.markersize = 4

        self.y_min = 10000
        
    def plot_new_figure(self):
        
        fig1, ax1 = plt.subplots(num = 1,figsize=(7.0,4.5))

        self.plot_from_existing_axes(ax1)
    
        plt.tight_layout(pad=0.0,h_pad=.30,rect=(0.02,0.01,0.99,0.98))

        if(self.is_save_plot):    
        
            str_file_name = cycle_label(self.dict_cycle_data, False, True) + "__PH_plot.png"
            
            if(self.file_name != ""):
                str_file_name = self.file_name + ".png"
                                       
            fig1.savefig('results/' + str_file_name)
            
            plt.close()

    def set_y_min(self):

        if (self.dict_cycle_data["cycle_config"] == 1):
            P_min = self.dict_cycle_data["P_state_points"][0]
        else:
            P_min = self.dict_cycle_data["P_state_points"][10]

        if(P_min > 5):
            P_min = floor_nearest_base(P_min, 5)
        else:
            P_min = floor_nearest_base(P_min, 0.5)

        self.y_min = min(self.y_min, P_min)

    def plot_from_existing_axes(self, ax_in):

        # eta_str = "Thermal Efficiency = " + '{:.1f}'.format(self.dict_cycle_data["eta_thermal_calc"]*100) + "%"
    
        plot_title = self.is_overwrite_title
        
        # if(self.dict_cycle_data["cycle_config"] == 1):
        #     if(self.is_overwrite_title == ""):
        #         plot_title = "Recompression Cycle: " + eta_str
        # else:
        #     if(self.is_overwrite_title == ""):
        #         plot_title = "Partial Cooling Cycle, " + eta_str

        if (self.is_overwrite_title == ""):
            plot_title = get_plot_name(self.dict_cycle_data)
                
        self.overlay_cycle_data(ax_in)
        
        if(self.is_annotate):
            self.annotate(ax_in)
        
        self.format_axes(ax_in, plot_title)
        
        return ax_in
    
    def overlay_cycle_data(self, ax_in):
        
        if(self.dict_cycle_data["cycle_config"] == 1):
            self.plot_RC_points_and_lines(ax_in)
        else:
            self.plot_PC_points_and_lines(ax_in)
            
        self.set_y_min()
    
    def plot_RC_points_and_lines(self, ax_in):
        
        self.plot_shared_points_and_lines(ax_in)

        P_states = self.dict_cycle_data["P_state_points"]
        h_states = self.dict_cycle_data["h_state_points"]
        
        "Main cooler"
        ax_in.plot([h_states[8],h_states[0]],[P_states[8],P_states[0]], self.lc)
    
        ax_in.plot(h_states[0:10], P_states[0:10], self.lc + self.mt, markersize = 4)
    
        f_recomp = self.dict_cycle_data["recomp_frac"]
            
        if(f_recomp > 0.01):
            P_rc_data = self.dict_cycle_data["P_rc_data"]
            h_rc_data = self.dict_cycle_data["h_rc_data"]
            ax_in.plot(h_rc_data, P_rc_data, self.lc)
        
        return ax_in
    
    def plot_PC_points_and_lines(self, ax_in):
        
        self.plot_shared_points_and_lines(ax_in)
        
        P_states = self.dict_cycle_data["P_state_points"]
        h_states = self.dict_cycle_data["h_state_points"]
        
        "Pre cooler"
        ax_in.plot([h_states[8],h_states[10]],[P_states[8],P_states[10]], self.lc)
        
        "Pre compressor"
        P_pc_data = self.dict_cycle_data["P_pc_data"]
        h_pc_data = self.dict_cycle_data["h_pc_data"]
        ax_in.plot(h_pc_data, P_pc_data, self.lc)
        
        "Main cooler"
        ax_in.plot([h_states[11],h_states[0]],[P_states[11],P_states[0]], self.lc)
    
        ax_in.plot(h_states, P_states, self.lc + self.mt, markersize = 4)
    
        f_recomp = self.dict_cycle_data["recomp_frac"]
            
        if(f_recomp > 0.01):
            P_rc_data = self.dict_cycle_data["P_rc_data"]
            h_rc_data = self.dict_cycle_data["h_rc_data"]
            ax_in.plot(h_rc_data, P_rc_data, self.lc)
        
        return ax_in
    
    def plot_shared_points_and_lines(self, ax_in):
        
        P_states = self.dict_cycle_data["P_state_points"]
        h_states = self.dict_cycle_data["h_state_points"]
        
        "Main compressor"
        P_mc_data = self.dict_cycle_data["P_mc_data"]
        h_mc_data = self.dict_cycle_data["h_mc_data"]
        ax_in.plot(h_mc_data, P_mc_data, self.lc)
        
        "LTR HP"
        ax_in.plot([h_states[1],h_states[2]],[P_states[1],P_states[2]], self.lc)
        
        "HTR HP"
        ax_in.plot([h_states[3],h_states[4]],[P_states[3],P_states[4]], self.lc)
        
        "PHX"
        ax_in.plot([h_states[4],h_states[5]],[P_states[4],P_states[5]], self.lc)
        
        "Turbine"
        P_t_data = self.dict_cycle_data["P_t_data"]
        h_t_data = self.dict_cycle_data["h_t_data"]
        ax_in.plot(h_t_data, P_t_data, self.lc)
        
        "HTR LP"
        ax_in.plot([h_states[6],h_states[7]],[P_states[6],P_states[7]], self.lc)
        
        "LTR LP"
        ax_in.plot([h_states[7],h_states[8]],[P_states[7],P_states[8]], self.lc)
        
        return ax_in
    
    def format_axes(self, ax_in, plot_title):
            
        #ax_in.margins(x=0.1)
        
        ax_in.autoscale()
        y_down, y_up = ax_in.get_ylim()
        x_low, x_high = ax_in.get_xlim()
        
        if(self.is_add_T_const_lines):
            self.plot_constT(ax_in)     # add_Ph_constT(ax_in)
        if(self.is_add_dome):
            self.plot_dome(ax_in)        # add_Ph_dome(ax_in)
        
        ax_in.grid(alpha=0.5,which='major')
        ax_in.grid(alpha=0.3,which='minor')
        
        deltaP_base = 5
        ax_in.set_ylim(self.y_min, ceil_nearest_base(y_up, deltaP_base))
        y_down, y_up = ax_in.get_ylim()
        
        deltah_base = 100
        x_min = ceil_nearest_base(x_low - deltah_base, deltah_base)
        x_max = ceil_nearest_base(x_high, deltah_base)
        ax_in.set_xlim(x_low, x_high)
        major_x_ticks = np.arange(x_min, x_max+1,deltah_base)
        ax_in.set_xticks(major_x_ticks)
        
        ax_in.set_ylabel("Pressure [MPa]", fontsize = 12)
        ax_in.set_xlabel("Enthalpy [kJ/kg]", fontsize = 12)
        if(self.is_add_title):
            ax_in.set_title(plot_title, fontsize = 14) 
            
    def plot_constT(self, ax_in):

        fileDir = os.path.dirname(os.path.abspath(__file__))
        
        T_data = pd.read_csv(fileDir + "/property_data/constantT_data.txt")
        T_vals = []
        for names in T_data.columns.values.tolist():
            if names.split("_")[1] not in T_vals:
                T_vals.append(names.split("_")[1])
        
        i_last = len(T_data["P_"+T_vals[0]].values)  - 1
        i_ann = int(0.95 * i_last)
    
        for vals in T_vals:
            ax_in.plot(T_data["h_"+vals].values, T_data["P_"+vals].values, '--', color = 'tab:purple', lw = 0.5, alpha = 0.65)

        for vals in T_vals:
            ann_local = ax_in.annotate(vals+"C",xy=[T_data["h_"+vals].values[i_ann], T_data["P_"+vals].values[i_ann]], color = 'tab:purple', ha="center", alpha = 0.65, fontsize = 8)
            ann_local.set_in_layout(False)

    def plot_dome(self, ax_in):
    
        fileDir = os.path.dirname(os.path.abspath(__file__))
        
        ph_dome_data = pd.read_csv(fileDir + "/property_data/Ph_dome_data.txt")
        ax_in.plot(ph_dome_data["h"], ph_dome_data["P"], 'k-', lw = 1, alpha = 0.4)
    
    def annotate(self, ax_in):

        m_dot_co2_full = self.dict_cycle_data["m_dot_co2_full"]
        f_recomp = self.dict_cycle_data["recomp_frac"]
        m_dot_mc = m_dot_co2_full * (1.0 - f_recomp)
        m_dot_rc = m_dot_co2_full * f_recomp
        
        mc_title = r'$\bf{Main}$' + " " + r'$\bf{Compressor}$'
        m_dot_text = "\n" + r'$\.m$' + " = " + '{:.1f}'.format(m_dot_mc) + " kg/s"
        W_dot_text = "\nPower = " + '{:.1f}'.format(self.dict_cycle_data["mc_W_dot"]) + " MW"
        isen_text = "\n" + r'$\eta_{isen}$' + " = " + '{:.3f}'.format(self.dict_cycle_data["eta_isen_mc"])        
        
        mc_text = mc_title + m_dot_text + W_dot_text + isen_text
        if(self.is_annotate_MC_stages):
            stages_text = "\nStages = " + '{:d}'.format(int(self.dict_cycle_data["mc_n_stages"]))
            l_type, d_type = py_sco2.get_entry_data_type(self.dict_cycle_data["mc_D"])
            d_text = "\nDiameters [m] ="
            t_text = "\nTip Speed [-] ="
            if(l_type == "single"):
                d_text = d_text + " " + '{:.2f}'.format(self.dict_cycle_data["mc_D"])
                t_text = t_text + " " + '{:.2f}'.format(self.dict_cycle_data["mc_tip_ratio_des"])
            elif(l_type == "list"):
                for i_d, d_s in enumerate(self.dict_cycle_data["mc_D"]):
                    if(i_d == 0):
                        if(len(self.dict_cycle_data["mc_D"]) > 1):
                            space = "\n   "
                        else:
                            space = " "
                    else:
                        space = ", "
                    d_text = d_text + space + '{:.2f}'.format(d_s)
                    t_text = t_text + space + '{:.2f}'.format(self.dict_cycle_data["mc_tip_ratio_des"][i_d])
            mc_text = mc_text + stages_text + d_text + t_text
        
        P_states = self.dict_cycle_data["P_state_points"]
        h_states = self.dict_cycle_data["h_state_points"]
    
        P_mc_avg = 0.5*(P_states[0] + P_states[1])
        h_mc_avg = 0.5*(h_states[0] + h_states[1])
        
        if(self.is_annotate_MC):
            ax_in.annotate(mc_text, xy=(h_mc_avg,P_mc_avg), va="center", ha="center",multialignment="left",
                           fontsize = 8,
                           bbox=dict(boxstyle="round", fc="w", pad = 0.5))

        is_pc = self.dict_cycle_data["cycle_config"] == 2

        if (is_pc):
            t_weight = 0.5
        else:
            t_weight = 0.25

        t_title = r'$\bf{Turbine}$'
        m_dot_text = "\n" + r'$\.m$' + " = " + '{:.1f}'.format(m_dot_co2_full) + " kg/s"
        W_dot_text = "\nPower = " + '{:.1f}'.format(self.dict_cycle_data["t_W_dot"]) + " MW"
        isen_text = "\n" + r'$\eta_{isen}$' + " = " + '{:.3f}'.format(self.dict_cycle_data["eta_isen_t"])
    
        t_text = t_title + m_dot_text + W_dot_text + isen_text
        
        P_t_avg = t_weight*P_states[5] + (1.0-t_weight)*P_states[6]
        h_t_avg = t_weight*h_states[5] + (1.0-t_weight)*h_states[6]
        
        if(self.is_annotate_T):
            ax_in.annotate(t_text, xy=(h_t_avg,P_t_avg), va="center", ha="center",multialignment="left",
                       fontsize = 8,
                       bbox=dict(boxstyle="round", fc="w", pad = 0.5))
        

                          
        if(is_pc):
            pc_title = r'$\bf{Pre}$' + " " + r'$\bf{Compressor}$'
            m_dot_text = "\n" + r'$\.m$' + " = " + '{:.1f}'.format(m_dot_co2_full) + " kg/s"
            W_dot_text = "\nPower = " + '{:.1f}'.format(self.dict_cycle_data["pc_W_dot"]) + " MW"
            isen_text = "\n" + r'$\eta_{isen}$' + " = " + '{:.3f}'.format(self.dict_cycle_data["eta_isen_rc"])
            
            pc_text = pc_title + m_dot_text + W_dot_text + isen_text
            
            P_pc_avg = 0.5*(P_states[10] + P_states[11])
            h_pc_avg = 0.5*(h_states[10] + h_states[11])
            
            h_pc_text = h_states[11] + 3*(h_states[11] - h_states[10])
            
            if(self.is_annotate_PC):
                ax_in.annotate(pc_text, xy=(h_pc_avg, P_pc_avg),
                               xytext=(h_pc_text, P_states[11]), va="center",
                               arrowprops = dict(arrowstyle="->", color = 'b', ls = '--', lw = 0.6),
                               fontsize = 8, bbox=dict(boxstyle="round", fc="w", pad = 0.5))
        
        if(f_recomp > 0.01):
            
            rc_title = r'$\bf{Re}$' + " " + r'$\bf{Compressor}$'
            m_dot_text = "\n" + r'$\.m$' + " = " + '{:.1f}'.format(m_dot_rc) + " kg/s"
            W_dot_text = "\nPower = " + '{:.1f}'.format(self.dict_cycle_data["rc_W_dot"]) + " MW"
            isen_text = "\n" + r'$\eta_{isen}$' + " = " + '{:.3f}'.format(self.dict_cycle_data["eta_isen_rc"])
                        
            rc_text = rc_title + m_dot_text + W_dot_text + isen_text
            if(self.is_annotate_RC_stages):
                stages_text = "\nStages = " + '{:d}'.format(int(self.dict_cycle_data["rc_n_stages"]))
                l_type, d_type = py_sco2.get_entry_data_type(self.dict_cycle_data["rc_D"])
                d_text = "\nDiameters [m] ="
                t_text = "\nTip Speed [-] ="
                if(l_type == "single"):
                    d_text = d_text + " " + '{:.2f}'.format(self.dict_cycle_data["rc_D"])
                    t_text = t_text + " " + '{:.2f}'.format(self.dict_cycle_data["rc_tip_ratio_des"])
                elif(l_type == "list"):
                    for i_d, d_s in enumerate(self.dict_cycle_data["rc_D"]):
                        if(i_d == 0):
                            if(len(self.dict_cycle_data["rc_D"]) > 1):
                                space = "\n   "
                            else:
                                space = " "
                        else:
                            space = ", "
                        d_text = d_text + space + '{:.2f}'.format(d_s)
                        t_text = t_text + space + '{:.2f}'.format(self.dict_cycle_data["rc_tip_ratio_des"][i_d])
                rc_text = rc_text + stages_text + d_text + t_text

            rc_weight = 0.75
            if(is_pc):
                P_rc_avg = rc_weight*P_states[9] + (1.0-rc_weight)*P_states[11]
                h_rc_avg = rc_weight*h_states[9] + (1.0-rc_weight)*h_states[11]
                h_rc_text = h_states[9] + (h_states[9] - h_states[11])
                
            else:
                P_rc_avg = rc_weight*P_states[9] + (1.0-rc_weight)*P_states[8]
                h_rc_avg = rc_weight*h_states[9] + (1.0-rc_weight)*h_states[8]
                h_rc_text = h_states[9] + (h_states[9] - h_states[8])
    
            if(self.is_annotate_RC):
                ax_in.annotate(rc_text, xy=(h_rc_avg, P_rc_avg),
                               xytext=(h_rc_text, P_rc_avg), va="center",
                               arrowprops = dict(arrowstyle="->", color = 'b', ls = '--', lw = 0.6),
                               fontsize = 8, bbox=dict(boxstyle="round", fc="w", pad = 0.5)) 
            
        return ax_in
    

class C_sco2_TS_PH_plot:
    
    def __init__(self, dict_cycle_data):
        self.dict_cycle_data = dict_cycle_data
        
        self.c_TS_plot = C_sco2_cycle_TS_plot(self.dict_cycle_data)
        self.c_PH_plot = C_sco2_cycle_PH_plot(self.dict_cycle_data)
        
        self.is_save_plot = False
        self.is_annotate = True
        self.file_name = ""
        self.align = "vert"
        self.is_overwrite_title = ""
        
    def plot_new_figure(self):
        
        self.update_subplot_class_data()
        
        if(self.align == "horiz"):
            fig1, a_ax = plt.subplots(ncols=2, num=11, figsize=(10.0, 5.))
        else:
            fig1, a_ax = plt.subplots(nrows = 2,num = 11,figsize=(7.0,8.0))
        
        self.c_PH_plot.is_add_title = False
        self.c_TS_plot.is_add_title = False
        self.plot_from_existing_axes(a_ax)

        plot_title = self.is_overwrite_title        
        if (self.is_overwrite_title == ""):
            plot_title = get_plot_name(self.dict_cycle_data)
        fig1.suptitle(plot_title, fontsize=14)

        plt.tight_layout(pad=0.0,h_pad=2.0,w_pad=0.5,rect=(0.02,0.01,0.99,0.94))
        
        if(self.is_save_plot):
            
            if(self.file_name == ""):
                file_name = cycle_label(self.dict_cycle_data, False, True) + "__TS_PH_plots"
            else:
                file_name = self.file_name
    
            file_name = file_name + ".png"
            
            fig1.savefig('SSC CSP API/results/' + file_name)
            
            plt.close()
            
    def plot_from_existing_axes(self, list_ax_in):
        
        self.update_subplot_class_data()
        
        self.c_TS_plot.plot_from_existing_axes(list_ax_in[0])
                
        self.c_PH_plot.plot_from_existing_axes(list_ax_in[1])
        
    def overlay_existing_axes(self, list_ax_in):
        
        self.update_subplot_class_data()
        
        self.c_TS_plot.overlay_cycle_data(list_ax_in[0])
        
        self.c_PH_plot.overlay_cycle_data(list_ax_in[1])
        
    def update_subplot_class_data(self):
        
        self.c_TS_plot.is_annotate = self.is_annotate
        self.c_PH_plot.is_annotate = self.is_annotate
        
 
class C_sco2_TS_PH_overlay_plot:

    def __init__(self, dict_cycle_data1, dict_cycle_data2):
        self.dict_cycle_data1 = dict_cycle_data1
        self.dict_cycle_data2 = dict_cycle_data2
        
        self.is_save_plot = False
        
    def plot_new_figure(self):
         
        fig1, a_ax = plt.subplots(nrows = 2,num = 1,figsize=(7.0,8.0))

        c_plot1 = C_sco2_TS_PH_plot(self.dict_cycle_data1)
        c_plot1.is_annotate = False
        c_plot1.c_TS_plot.lc = 'k'
        c_plot1.c_TS_plot.mt = 's'
        c_plot1.c_PH_plot.lc = 'k'
        c_plot1.c_PH_plot.mt = 's'
        c_plot1.overlay_existing_axes(a_ax)

        ts_legend_lines = []
        ph_legend_lines = []
    
        ts_legend_lines.append(cycle_comp_legend_line(self.dict_cycle_data1, 'k', 's'))
        ph_legend_lines.append(cycle_comp_legend_line(self.dict_cycle_data1, 'k', 's', True))
        
        
        c_plot2 = C_sco2_TS_PH_plot(self.dict_cycle_data2)
        c_plot2.is_annotate = False
        c_plot2.c_TS_plot.lc = 'b'
        c_plot2.c_PH_plot.lc = 'b'
        c_plot2.overlay_existing_axes(a_ax)
         
        ts_legend_lines.append(cycle_comp_legend_line(self.dict_cycle_data2, 'b', 'o'))
        ph_legend_lines.append(cycle_comp_legend_line(self.dict_cycle_data2, 'b', 'o', True))
                   
        c_plot1.c_TS_plot.format_axes(a_ax[0], "Cycle Comparison")
        c_plot1.c_PH_plot.is_add_title = False
        c_plot1.c_PH_plot.format_axes(a_ax[1], "")
         
        a_ax[0].legend(handles=ts_legend_lines, fontsize = 8)
        a_ax[1].legend(handles=ph_legend_lines, labelspacing = 1.0, loc = "center", fontsize = 8, borderpad = 1)
         
        plt.tight_layout(pad=0.0,h_pad=2.0,rect=(0.02,0.01,0.99,0.96))
        
        if(self.is_save_plot):
            
            txt_label_1 = cycle_label(self.dict_cycle_data1, False, True)
    
            txt_label_2 = cycle_label(self.dict_cycle_data2, False, True)
            
            file_name = txt_label_1 + "__vs__" + txt_label_2 + ".png"
            
            fig1.savefig('results/' + file_name)
            
            plt.close()


def custom_auto_y_axis_scale(axis_in):
    
    y_lower, y_upper = axis_in.get_ylim()
    mult = 1
    if(y_upper < 0.1):
        mult = 100
    elif(y_upper < 2):
        mult = 100
    elif(y_upper < 10):
        mult = 10
    base = np.ceil((mult*y_upper - mult*y_lower)*0.5)
    
    y_lower_new = (1/mult)*base*np.floor(mult*y_lower*0.99/base)
    y_upper_new = (1/mult)*base*np.ceil(mult*y_upper*1.01/base)
    
    #print("base = ", base)
    #print("y_lower = ", y_lower, "y_upper = ", y_upper)
    #print("y_lower_new = ", y_lower_new, "y_upper_new = ", y_upper_new)
    
    axis_in.set_ylim(y_lower_new, y_upper_new)
        
def cycle_comp_legend_line(cycle_data, color, marker, is_multi_line = False):

    label = cycle_label(cycle_data, is_multi_line, False)
                                                          
    return mlines.Line2D([],[],color = color, marker = marker, label = label)

def cycle_label(cycle_data, is_multi_line = False, is_file_name = False):
    
    if(cycle_data["cycle_config"] == 2):
        cycle_name = r'$\bf{Partial}$' + " " + r'$\bf{Cooling}$'
        cycle_abv = "PC"
    elif cycle_data["cycle_config"] == 1 and cycle_data["is_recomp_ok"] == 1:
        cycle_name = r'$\bf{Recompression}$'
        cycle_abv = "RC"
    else:
        cycle_name = r'$\bf{Simple}$'
        cycle_abv = "simple"
    
    if(is_multi_line):
        label = cycle_name + ": " + "\n" + r'$\eta$' + " = " + '{:.1f}'.format(cycle_data["eta_thermal_calc"]*100) + "%" #,\nUA = "+ '{:.1f}'.format(cycle_data["UA_recup_total"]) + " MW/K"
    elif(not(is_file_name)):
        label = cycle_name + ": " + r'$\eta$' + " = " + '{:.1f}'.format(cycle_data["eta_thermal_calc"]*100) + "%" #, UA = "+ '{:.1f}'.format(cycle_data["UA_recup_total"]) + " MW/K"
    else:
        label = cycle_abv + "_eta_"+ '{:.1f}'.format(cycle_data["eta_thermal_calc"]*100)
        #label = cycle_abv + "_UA_"+ '{:.1f}'.format(cycle_data["UA_recup_total"]) + "_eta_"+ '{:.1f}'.format(cycle_data["eta_thermal_calc"]*100)

    return label

class C_OD_stacked_outputs_plot:
    
    def __init__(self, list_dict_results):
        
        self.list_dict_results = list_dict_results
        
        self.is_save = False
        self.file_name = ""
        self.dpi = 300
        self.file_ext = ".png"
        
        self.x_var = "T_amb"
        
        self.is_legend = True
        self.leg_var = "T_amb"
        self.list_leg_spec = ""
        
        self.add_subplot_letter = False
                
        self.is_separate_design_point = False       # Separates design point in legend
        self.is_plot_each_des_pt = True             # if False, assumes each dict dataset has same design point
        self.is_plot_des_pts = True                 # if True, plot marker for design point
        self.list_des_pts = [0]                      # if is_plot_des_pts is False, then this list can specify which datasets' design points are plotted. e.g. [0,3]
        
        self.is_shade_infeasible = False             # for FINAL dataset, shade all subplot where any metric in y_val exceeds its value in var_info_metrics

        self.y_vars = ["eta", "W_dot"]
        self.shade_if_infeasible_var = ""          # Set to ["var_name"] to shade
        
        self.plot_colors = ['k','b','g','r','c','y']
        self.l_s = ['-','--','-.',':']
        self.d_s = [[6,0],[1,1],[3,2],[1,2,1,2,1,4],[4,2,4,2,1,2],[2,1,0.5,1,0.5,1]]                   #[4,2,1,2,1,2]]              #[3,6,3,6,3,18],[12,6,12,6,3,6],[12,6,3,6,3,6]]
        self.is_linestyle_not_dash = True
        self.mss = ["o","d","s",'^','p']
        self.mrk_sz = 6
        self.l_w = 1
        self.is_line = True                        # True: correct points with line, no markers, False: plot points with markers, no lines
        self.is_change_ls_each_plot = True          # True: every dataset cycle line style; False: change line style after all colors cycle
        
        self.is_x_label_long = True

        self.is_spec_subplot_dimensions = True
        self.h_subplot = 3.0            # Used if 'is_spec_subplot_dimensions' = True
        self.w_subplot = 4.0            # Used if 'is_spec_subplot_dimensions' = True
        self.h_figure = 7.48            # Used if 'is_spec_subplot_dimensions' = False
        self.w_figure = 7.48            # Used if 'is_spec_subplot_dimensions' = False
        
        self.axis_label_fontsize = 14        #8
        self.tick_lab_fontsize = 12         #8
        self.legend_fontsize = 14           #8
        
        self.n_leg_cols = 3
        self.is_label_leg_cols = ""        # LIST
        
        self.bb_y_max_is_leg = 0.92
        self.bb_h_pad = 2
        self.bb_w_pad = 0.75
        
        self.max_rows = 3

        self.var_info_metrics = py_sco2.get_des_od_label_unit_info__combined()
        self.var_results_check = "eta_thermal_od"
        self.fig_num = 1
        
    def create_plot(self):
        
        # May help to print this if plotting code is failing
        #print(self.list_dict_results[0][self.var_results_check])
        
        legend_lines = []
        legend_labels = []
        
        self.x_var_des = self.var_info_metrics[self.x_var].des_var
        self.x_var_od = self.var_info_metrics[self.x_var].od_var
                                             
        if(self.is_x_label_long):
            self.x_label = self.var_info_metrics[self.x_var].l_label
        else:
            self.x_label = self.var_info_metrics[self.x_var].s_label
        
        n_subplots = len(self.y_vars)

        n_cols = (n_subplots - 1)//self.max_rows + 1
        #print("Columns = ", n_cols)
        n_rows = int(np.ceil(n_subplots/n_cols))
        #print("Rows = ", n_rows)

        if(self.is_spec_subplot_dimensions):
            f_h = self.h_subplot * n_rows
            f_w = self.w_subplot * n_cols
        else:
            f_h = self.h_figure
            f_w = self.w_figure

        fig1, a_ax = plt.subplots(nrows = n_rows, ncols = n_cols, num = self.fig_num,figsize=(f_w,f_h))
        
        n_datasets = len(self.list_dict_results)
        
        n_leg_rows = n_datasets / self.n_leg_cols

        for i in range(n_datasets):
        
            n_od_pts_i = len(self.list_dict_results[i][self.x_var_od])
            y_feasible_flag_i = [False for i in range(n_od_pts_i)]      # This is reset every dataset...
            
            color_od_i = self.plot_colors[i%len(self.plot_colors)]
            if(self.is_change_ls_each_plot):
                ds_od_i = self.d_s[i%len(self.d_s)]
                ls_od_i = self.l_s[i%len(self.l_s)]
            else:
                color_iteration = i // len(self.plot_colors)
                ls_od_i = self.l_s[color_iteration%len(self.l_s)]
                ds_od_i = self.l_s[color_iteration%len(self.d_s)]
            mrk_i = self.mss[i%len(self.mss)]
            
            ls_des_i = color_od_i + mrk_i
        
            if(self.var_info_metrics[self.x_var].des == -999):                                                                                                                           
                x_val_des_i = self.list_dict_results[i][self.x_var_des]  # Design point x value
            else:
                x_val_des_i = self.var_info_metrics[self.x_var].des      # Design point x value
                 
            des_txt_leg_i = ""
            od_txt_leg_i = ""
                                                 
            if(self.list_leg_spec != ""):
                des_txt_leg_i = self.list_leg_spec[i]
                if(not(isinstance(self.list_leg_spec[i], list))):
                    self.is_separate_design_point = False
            else:
                if(self.var_info_metrics[self.leg_var].des == -999):
                    des_leg_val_i = self.list_dict_results[i][self.var_info_metrics[self.leg_var].des_var]
                else:
                    des_leg_val_i = self.var_info_metrics[self.leg_var].des
                
                od_leg_val_i = self.list_dict_results[i][self.var_info_metrics[self.leg_var].od_var][0]
                
                des_txt_leg_i = "Design " + self.var_info_metrics[self.leg_var].l_label + " = " + "{:.2f}".format(des_leg_val_i)
                od_txt_leg_i = self.var_info_metrics[self.leg_var].l_label + " = " + "{:.2f}".format(od_leg_val_i)
            
            if( self.is_label_leg_cols != "" and len(self.is_label_leg_cols) > 1 and i % n_leg_rows == 0 ):
                if(len(self.is_label_leg_cols) == self.n_leg_cols):
                    legend_lines.append(mlines.Line2D([],[], color = 'w'))
                    legend_labels.append(self.is_label_leg_cols[int(i // n_leg_rows)])
                else:
                    print("The number of input Legend Column Labels", len(self.is_label_leg_cols),
                          "is not equal to the number of legend columns", self.n_leg_cols)

            mrk_i_leg = ""
            if(self.is_separate_design_point):
                if(self.is_plot_each_des_pt or i == 0):
                        # Marker
                    legend_lines.append(mlines.Line2D([],[],color = color_od_i, marker = mrk_i_leg,  label = des_txt_leg_i))
                    legend_labels.append(des_txt_leg_i)
                    # Line
                if(self.is_linestyle_not_dash):
                    legend_lines.append(mlines.Line2D([],[],color = color_od_i, ls = ls_od_i, label = od_txt_leg_i))
                else:
                    legend_lines.append(mlines.Line2D([], [], color=color_od_i, dashes=ds_od_i, label=od_txt_leg_i))
                legend_labels.append(od_txt_leg_i)                   
            else:
                    # Marker & Line
                if(des_txt_leg_i != ""):
                    if(self.is_linestyle_not_dash):
                        legend_lines.append(mlines.Line2D([],[],color = color_od_i, ls = ls_od_i, marker = mrk_i_leg,  label = des_txt_leg_i))
                    else:
                        legend_lines.append(mlines.Line2D([], [], color=color_od_i, dashes=ds_od_i, marker=mrk_i_leg, label=des_txt_leg_i))
                    legend_labels.append(des_txt_leg_i)

            #mlines.Line2D([], [], color=color_od_i, dashes=[3,6,3,6,3,18], marker=mrk_i, label=des_txt_leg_i)

            if not(self.is_line):
                ls_od_i = mrk_i
                ds_od_i = mrk_i


            if(self.shade_if_infeasible_var != ""):

                x_infeasible_ranges = []

                for jj, j_key in enumerate(self.shade_if_infeasible_var):

                    jj_y_limit = self.var_info_metrics[j_key].limit_var
                    j_y_od_key = self.var_info_metrics[j_key].od_var

                    if (jj_y_limit != ""):

                        if (isinstance(jj_y_limit, str)):
                            ii_y_limit = self.list_dict_results[i][jj_y_limit]
                        else:
                            ii_y_limit = jj_y_limit


                        is_j_prev_infeas = False
                        is_j_infeas = False

                        for j_in, y_val_local in enumerate(self.list_dict_results[i][j_y_od_key]):

                            is_j_prev_infeas = is_j_infeas
                            is_j_infeas = False

                            if (isinstance(y_val_local, list)):
                                for k_in, y_k_val_local in enumerate(y_val_local):
                                    if (self.var_info_metrics[j_key].limit_var_type == "max"):
                                        if (y_k_val_local > ii_y_limit):
                                            is_j_infeas = True

                                    else:
                                        if (y_k_val_local < ii_y_limit):
                                            is_j_infeas = True

                            else:
                                if (self.var_info_metrics[j_key].limit_var_type == "max"):
                                    if (y_val_local > ii_y_limit):
                                        is_j_infeas = True

                                else:
                                    if (y_val_local < ii_y_limit):
                                        is_j_infeas = True

                            if(not(is_j_prev_infeas) and not(is_j_infeas)):

                                a = 1

                            elif(not(is_j_prev_infeas) and is_j_infeas):

                                j_low = max(j_in-1,0)

                            elif(is_j_prev_infeas and not(is_j_infeas)):

                                j_high = j_in
                                x_infeasible_ranges.append([j_low,j_high])

                        if(is_j_infeas):
                            j_high = j_in
                            x_infeasible_ranges.append(j_low,j_high)


            for j, key in enumerate(self.y_vars):
                
                j_l_i = string.ascii_lowercase[j%26]
                
                j_col = j//n_rows
                j_row = j%n_rows

                y_od_key = self.var_info_metrics[key].od_var
                y_des_key = self.var_info_metrics[key].des_var
                y_label = self.var_info_metrics[key].s_label
                y_limit = self.var_info_metrics[key].limit_var
                
                if(self.add_subplot_letter):
                    y_label = r'$\bf{' + format(j_l_i) + ")" + '}$' + " " + y_label
                        
                if(n_cols > 1 and n_rows > 1):
                    j_axis = a_ax[j_row,j_col]
                elif(n_rows > 1):
                    j_axis = a_ax[j_row]
                elif(n_cols > 1):
                    j_axis = a_ax[j_col]
                else:
                    j_axis = a_ax

                if(self.is_linestyle_not_dash):
                    j_axis.plot(self.list_dict_results[i][self.x_var_od], self.list_dict_results[i][y_od_key],
                                color_od_i+ls_od_i, markersize = self.mrk_sz, linewidth = self.l_w)
                else:
                    j_axis.plot(self.list_dict_results[i][self.x_var_od], self.list_dict_results[i][y_od_key],
                            color_od_i, dashes=ds_od_i, markersize=self.mrk_sz, linewidth = self.l_w)

                if (self.shade_if_infeasible_var != ""):

                    if(len(x_infeasible_ranges)>0):

                        for i_x_range in x_infeasible_ranges:

                            if (self.is_linestyle_not_dash):
                                j_axis.plot(self.list_dict_results[i][self.x_var_od][i_x_range[0]:i_x_range[1]],
                                            self.list_dict_results[i][y_od_key][i_x_range[0]:i_x_range[1]],
                                            color_od_i + ls_od_i, markersize=self.mrk_sz, linewidth=self.l_w*6,alpha=0.5)
                            else:
                                j_axis.plot(self.list_dict_results[i][self.x_var_od][i_x_range[0]:i_x_range[1]],
                                            self.list_dict_results[i][y_od_key][i_x_range[0]:i_x_range[1]],
                                            color_od_i, dashes=ds_od_i, markersize=self.mrk_sz, linewidth=self.l_w*6,alpha=0.5)
                
                if(self.is_plot_des_pts or (i in self.list_des_pts)):
                    if(self.is_plot_each_des_pt or (self.is_plot_des_pts and i == 0) or (i in self.list_des_pts)):
                        if(self.var_info_metrics[key].des_d_type == "single"):
                            if(self.var_info_metrics[key].des_var == "none"):
                                j_axis.plot(x_val_des_i, self.var_info_metrics[key].des, ls_des_i, markersize = self.mrk_sz)
                            else:
                                j_axis.plot(x_val_des_i, self.list_dict_results[i][y_des_key], ls_des_i, markersize = self.mrk_sz)
                        elif(self.var_info_metrics[key].des_d_type == "list"):
                            for i_s in range(len(self.list_dict_results[i][y_des_key])):
                                j_axis.plot(x_val_des_i, self.list_dict_results[i][y_des_key][i_s], ls_des_i, markersize = self.mrk_sz)
                                
                if(y_limit != ""):
                    if(isinstance(y_limit, str)):
                        y_limit_list = [self.list_dict_results[i][y_limit] for ind in range(len(self.list_dict_results[i][self.x_var_od]))]
                    else:
                        y_limit_list = [y_limit for ind in range(len(self.list_dict_results[i][self.x_var_od]))]

                    j_axis.plot(self.list_dict_results[i][self.x_var_od], y_limit_list, 'm:', linewidth = self.l_w)
                    
                if(i == n_datasets - 1):
                    
                    j_axis.set_ylabel(y_label, fontsize = self.axis_label_fontsize)
                    
                    if(self.var_info_metrics[key].y_label_style == "sci"):
                        j_axis.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
                        j_axis.yaxis.get_offset_text().set_fontsize(self.tick_lab_fontsize)
                
                    j_axis.tick_params(labelsize = self.tick_lab_fontsize)
                    
                    if( self.var_info_metrics[self.x_var].y_axis_min_max != "" ):
                        j_axis.set_xlim(self.var_info_metrics[self.x_var].y_axis_min_max[0], self.var_info_metrics[self.x_var].y_axis_min_max[1])
                        
                        if( self.var_info_metrics[self.x_var].ticks != "" ):
                            j_axis.set_xticks(self.var_info_metrics[self.x_var].ticks)
                            
                    if( self.var_info_metrics[self.x_var].minloc != "" ):
                        j_axis.xaxis.set_minor_locator(AutoMinorLocator(self.var_info_metrics[self.x_var].minloc))
                
                    j_axis.grid(which = 'both', color = 'gray', alpha = 0.5)
                
                    if(j_row == n_rows - 1 or j == n_subplots - 1):
                        j_axis.set_xlabel(self.x_label, fontsize = self.axis_label_fontsize)
                        
                    if(self.var_info_metrics[key].y_axis_min_max == ""):
                        custom_auto_y_axis_scale(j_axis)
                    else:
                        j_axis.set_ylim(self.var_info_metrics[key].y_axis_min_max[0], self.var_info_metrics[key].y_axis_min_max[1])
                        
                    if(self.var_info_metrics[key].minloc != ""):
                        j_axis.yaxis.set_minor_locator(AutoMinorLocator(self.var_info_metrics[key].minloc))

                        
                if(self.is_shade_infeasible):
                    if(y_limit != ""):
                        if (isinstance(y_limit, str)):
                            i_y_limit = self.list_dict_results[i][y_limit]
                        else:
                            i_y_limit = y_limit
                        for j_in, y_val_local in enumerate(self.list_dict_results[i][y_od_key]):
                            if(isinstance(y_val_local,list)):
                               for k_in, y_k_val_local in enumerate(y_val_local):
                                   if(self.var_info_metrics[key].limit_var_type == "max"):
                                       if(y_k_val_local > i_y_limit):
                                           y_feasible_flag_i[j_in] = True
                                   else:
                                       if(y_k_val_local < i_y_limit):
                                           y_feasible_flag_i[j_in] = True
                            else:
                                if(self.var_info_metrics[key].limit_var_type == "max"):
                                    if(y_val_local > i_y_limit):
                                        y_feasible_flag_i[j_in] = True
                                else:
                                    if(y_val_local < i_y_limit):
                                        y_feasible_flag_i[j_in] = True

        #fig1.legend(legend_lines, legend_labels, fontsize = self.legend_fontsize, ncol = self.n_leg_cols, 
        #     loc = "upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0))        
        
        if(self.is_legend):
            if( self.is_label_leg_cols != "" and len(self.is_label_leg_cols) == 1):
                ii_leg = fig1.legend(legend_lines, legend_labels, title = self.is_label_leg_cols[0], fontsize = self.legend_fontsize, ncol = self.n_leg_cols, 
                     loc = "upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0))
                plt.setp(ii_leg.get_title(),fontsize=self.legend_fontsize)
            else:
                fig1.legend(legend_lines, legend_labels, fontsize = self.legend_fontsize, ncol = self.n_leg_cols, 
                     loc = "upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0))
        
        if(self.is_legend):
            plt.tight_layout(pad=0.0,h_pad=self.bb_h_pad, w_pad = self.bb_w_pad, rect=(0.012,0.02,0.98,self.bb_y_max_is_leg))
        else:
            plt.tight_layout(pad=0.0,h_pad=self.bb_h_pad, w_pad = self.bb_w_pad, rect=(0.02,0.02,0.99,0.96))
        
        # Hide unused subplots
        for j in range(n_subplots, n_cols*n_rows):
            j_col = j//n_rows
            j_row = j%n_rows
            
            a_ax[j_row,j_col].set_visible(False)
        
        # Shade infeasible regions
        for j, key in enumerate(self.y_vars):
            j_col = j//n_rows
            j_row = j%n_rows

            if (n_cols > 1 and n_rows > 1):
                j_axis = a_ax[j_row, j_col]
            elif (n_rows > 1):
                j_axis = a_ax[j_row]
            elif (n_cols > 1):
                j_axis = a_ax[j_col]
            else:
                j_axis = a_ax

            y_lower, y_upper = j_axis.get_ylim()
            j_axis.fill_between(self.list_dict_results[0][self.x_var_od], y_lower, y_upper, where=y_feasible_flag_i, facecolor='red', alpha=0.5)
           
        if(self.is_save and self.file_name != ""):    
         
            plt.savefig('results/' + self.file_name + self.file_ext, dpi = self.dpi)
    
            plt.close() 

class C_des_stacked_outputs_plot:
    
    def __init__(self, list_dict_results):
        
        self.list_dict_results = list_dict_results
        
        self.is_save = False
        self.file_name = ""
        self.dpi = 300
        self.file_ext = ".png"
        
        self.x_var = "recup_tot_UA"
        
        self.is_legend = True
        self.leg_var = "min_phx_deltaT"     # variable that legend uses to differentiate datasets. overwritten if list_leg_spec is defined
        self.list_leg_spec = ""             # list of strings for legend for each dataset
        
        self.add_subplot_letter = False

        #### component limits - Partial Cooling (lots of plots)
        self.y_vars = ["eta","cycle_cost"]
        
        self.min_var = ""             # If this is defined, plot will add a point at the x value corresponding to the min value of this variable
        
        self.plot_colors = ['k','b','g','r','c']
        self.l_s = ['-','--','-.',':']
        self.mss = ["o","d","s",'^','p']
        self.is_line_mkr = True
        self.is_change_ls_each_plot = True          # True: every dataset cycle line style; False: change line style after all colors cycle
        
        self.is_x_label_long = True
        self.h_subplot = 3.0
        self.w_subplot = 4.0
        
        self.axis_label_fontsize = 14        #8
        self.tick_lab_fontsize = 12         #8
        self.legend_fontsize = 14           #8
        
        self.n_leg_cols = 3
        self.is_label_leg_cols = ""
        
        self.bb_y_max_is_leg = 0.92
        self.bb_h_pad = 2
        self.bb_w_pad = 0.75
        
        self.max_rows = 3

        self.var_info_metrics = py_sco2.get_des_od_label_unit_info__combined()
        self.var_results_check = "eta_thermal_calc"
        self.fig_num = 1
        
    def set_var_info_metrics_to_sam_mspt(self):
        
        self.var_info_metrics = py_sco2.get_sam_mspt_sco2_label_unit_info()
        self.var_results_check = "annual_cycle_output"
        
    def set_var_inf_metrics_to_sco2_mspt_combined(self):
        
        self.var_info_metrics = py_sco2.get_des_od_mspt_label_unit_info__combined()
        
    def create_plot(self):
        
        #print(self.list_dict_results[0][self.var_results_check])
        
        legend_lines = []
        legend_labels = []
        
        self.x_var_des = self.var_info_metrics[self.x_var].des_var
        
        if(self.is_x_label_long):
            self.x_label = self.var_info_metrics[self.x_var].l_label
        else:
            self.x_label = self.var_info_metrics[self.x_var].s_label
                                             
        n_subplots = len(self.y_vars)
        
        n_cols = (n_subplots - 1)//self.max_rows + 1
        #print("Columns = ", n_cols)
        n_rows = int(np.ceil(n_subplots/n_cols))
        #print("Rows = ", n_rows)
        
        f_h = self.h_subplot * n_rows
        f_w = self.w_subplot * n_cols
        
        fig1, a_ax = plt.subplots(nrows = n_rows, ncols = n_cols, num = self.fig_num,figsize=(f_w,f_h))
        
        n_datasets = len(self.list_dict_results)
        
        n_leg_rows = n_datasets / self.n_leg_cols
        
        for i in range(n_datasets):

            if(self.min_var != ""):
                i_min_var = self.list_dict_results[i][self.min_var].index(min(self.list_dict_results[i][self.min_var]))
            
            color_i = self.plot_colors[i%len(self.plot_colors)]
            if(self.is_change_ls_each_plot):
                ls_i = self.l_s[i%len(self.l_s)]
            else:
                color_iteration = i // len(self.plot_colors)
                ls_i = self.l_s[color_iteration%len(self.l_s)]
            mrk_i = self.mss[i%len(self.mss)]
            clr_mrk_i = color_i + mrk_i
            
            if(self.is_line_mkr):
                plt_style_i = ls_i + clr_mrk_i
            else:
                plt_style_i = color_i + ls_i
            
            i_txt_leg = ""
            
            if(self.list_leg_spec != ""):
                if(i < len(self.list_leg_spec)):
                    i_txt_leg = self.list_leg_spec[i]
            else:
                des_leg_val = self.list_dict_results[i][self.var_info_metrics[self.leg_var].des_var][0]
                i_txt_leg = self.var_info_metrics[self.leg_var].l_label + " = " + "{:2f}".format(des_leg_val)
            
            if( self.is_label_leg_cols != "" and i % n_leg_rows == 0 ):
                if(len(self.is_label_leg_cols) == self.n_leg_cols):
                    legend_lines.append(mlines.Line2D([],[], color = 'w'))
                    legend_labels.append(self.is_label_leg_cols[int(i // n_leg_rows)])                
                #else:
                #    print("The number of input Legend Column Labels", len(self.is_label_leg_cols),
        		#		  "is not equal to the number of legend columns", self.n_leg_cols)
                        
            if(self.is_line_mkr):
                legend_lines.append(mlines.Line2D([],[],color = color_i, ls = ls_i, marker = mrk_i,  label = i_txt_leg))
            else:
                legend_lines.append(mlines.Line2D([],[],color = color_i, ls = ls_i, label = i_txt_leg))
            legend_labels.append(i_txt_leg)
            
            for j, key in enumerate(self.y_vars):
                
                j_l_i = string.ascii_lowercase[j]
                
                j_col = j//n_rows
                j_row = j%n_rows
                
                y_des_key = self.var_info_metrics[key].des_var
                y_label = self.var_info_metrics[key].s_label
                                               
                if(self.add_subplot_letter):
                    y_label = r'$\bf{' + format(j_l_i) + ")" + '}$' + " " + y_label
                        
                if(n_cols > 1):
                    j_axis = a_ax[j_row,j_col]
                elif(n_rows > 1):
                    j_axis = a_ax[j_row]
                else:
                    j_axis = a_ax
                    
                j_axis.plot(self.list_dict_results[i][self.x_var_des], self.list_dict_results[i][y_des_key], plt_style_i)
                
                if(self.min_var != ""):
                    j_axis.plot(self.list_dict_results[i][self.x_var_des][i_min_var], self.list_dict_results[i][y_des_key][i_min_var], 
                                clr_mrk_i, markersize=self.tick_lab_fontsize*0.65)
                
                if(i == n_datasets - 1):
                    j_axis.set_ylabel(y_label, fontsize = self.axis_label_fontsize)
                    
                    if(self.var_info_metrics[key].y_label_style == "sci"):
                        j_axis.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
                        j_axis.yaxis.get_offset_text().set_fontsize(self.tick_lab_fontsize)
                    
                    j_axis.tick_params(labelsize = self.tick_lab_fontsize)
                    
                    if( self.var_info_metrics[self.x_var].y_axis_min_max != "" ):
                        j_axis.set_xlim(self.var_info_metrics[self.x_var].y_axis_min_max[0], self.var_info_metrics[self.x_var].y_axis_min_max[1])
                        
                        if( self.var_info_metrics[self.x_var].ticks != "" ):
                            j_axis.set_xticks(self.var_info_metrics[self.x_var].ticks)
                            
                        if( self.var_info_metrics[self.x_var].minloc != "" ):
                            j_axis.xaxis.set_minor_locator(AutoMinorLocator(self.var_info_metrics[self.x_var].minloc)) #set_minor_locator(MultipleLocator(self.var_info_metrics[self.x_var].minloc))
                    
                    j_axis.grid(which = 'both', color = 'gray', alpha = 0.5)
                    
                    if(j_row == n_rows - 1 or j == n_subplots - 1):
                        j_axis.set_xlabel(self.x_label, fontsize = self.axis_label_fontsize)
                        
                        
                    if(self.var_info_metrics[key].y_axis_min_max == ""):
                        custom_auto_y_axis_scale(j_axis)
                    else:
                        j_axis.set_ylim(self.var_info_metrics[key].y_axis_min_max[0], self.var_info_metrics[key].y_axis_min_max[1])
                        
                    if(self.var_info_metrics[key].minloc != ""):
                        j_axis.yaxis.set_minor_locator(AutoMinorLocator(self.var_info_metrics[key].minloc))

        if(self.is_legend):
            if( self.is_label_leg_cols != "" and len(self.is_label_leg_cols) == 1):
                ii_leg = fig1.legend(legend_lines, legend_labels, title = self.is_label_leg_cols[0], fontsize = self.legend_fontsize, ncol = self.n_leg_cols, 
                     loc = "upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0))
                plt.setp(ii_leg.get_title(),fontsize=self.legend_fontsize)
            else:
                fig1.legend(legend_lines, legend_labels, fontsize = self.legend_fontsize, ncol = self.n_leg_cols, 
                     loc = "upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0))
        
        if(self.is_legend):
            plt.tight_layout(pad=0.0,h_pad=self.bb_h_pad, w_pad = self.bb_w_pad, rect=(0.012,0.02,0.98,self.bb_y_max_is_leg))
        else:
            plt.tight_layout(pad=0.0,h_pad=self.bb_h_pad, w_pad = self.bb_w_pad, rect=(0.02,0.02,0.99,0.96))
        
        # Hide unused subplots
        for j in range(n_subplots, n_cols*n_rows):
            j_col = j//n_rows
            j_row = j%n_rows
            a_ax[j_row,j_col].set_visible(False)

        if(self.is_save and self.file_name != ""):
            plt.savefig('results/' + self.file_name + self.file_ext, dpi = self.dpi)
            plt.close() 
                
        return

class C_stacked_cycle_outputs_comp_plot:

    def __init__(self, cycle_config, data_1, data_2, x_var, x_label, y_var_dict_ordered, 
                 plt_title, save_title,
                 data_1_desc = "", data_2_desc = ""):
        self.cycle_config = cycle_config
        self.data_1 = data_1
        self.data_1_desc = data_1_desc
        self.data_2 = data_2
        self.data_2_desc = data_2_desc
        self.x_var = x_var
        self.x_label = x_label
        self.y_var_dict_ordered = y_var_dict_ordered
                
        if(self.data_1_desc == ""):
            self.data_1_desc = "data_set_1"
            
        if(self.data_2_desc == ""):
            self.data_2_desc = "data_set_2"
            
        self.plt_title = plt_title
        
        self.save_title = save_title

    def create_stacked_cycle_output_comp(self):

        y_keys = list(self.y_var_dict_ordered.keys())
        print("Ordered y-vars to plot = ", y_keys)
  
        n_subplots = len(y_keys)
        
        h = 2.0*n_subplots
        fig1, a_ax = plt.subplots(nrows = n_subplots,num = 1,figsize=(7.0,h))
        
        for j in range(n_subplots):

            y_key = y_keys[j]
            y_label = self.y_var_dict_ordered[y_key]
            
            a_ax[j].plot(self.data_1[self.x_var], self.data_1[y_key], label = self.data_1_desc)
            a_ax[j].plot(self.data_2[self.x_var], self.data_2[y_key], label = self.data_2_desc)
            a_ax[j].set_ylabel(y_label)
            
            a_ax[j].grid(which = 'both', color = 'gray', alpha = 1)
            
            x = 0.9*a_ax[j].get_xlim()[1] + 0.1*a_ax[j].get_xlim()[0]
            y = 0.5*a_ax[j].get_ylim()[0] + 0.5*a_ax[j].get_ylim()[1]
            letter = string.ascii_lowercase[j]
            
            a_ax[j].text(x, y, letter, bbox = dict(facecolor = 'w', edgecolor = 'k'))
    
            if( j == 0 ):
                a_ax[j].set_title(self.plt_title)
                a_ax[j].legend(loc = 'center', bbox_to_anchor=(0.5,1.35),ncol=2)
                #mlines.Line2D([],[],color = color, marker = marker, label = label)
    
            if( j == n_subplots - 1 ):
                
                a_ax[j].set_xlabel(self.x_label)
        
        #plt.tight_layout(pad=0.0,h_pad=.30,rect=(0.02,0.01,0.99,0.98))
        plt.tight_layout(pad=0.0,h_pad=2.0,rect=(0.02,0.01,0.99,0.965))
        
        plt.savefig('results/' + self.save_title + ".png")
        
        plt.close()        
        
class C_UA_par__stacked_outputs_comp_plot(C_stacked_cycle_outputs_comp_plot):
    
    def __init__(self, cycle_config, data_1, data_2, data_desc_1 = "", data_desc_2  = ""):
        
        x_var = "UA_recup_tot_des"
        x_label = "Total recuperator conductance [kW/K]"
        y_var_dict_ordered = {"eta_thermal_calc" : "Thermal\nEfficiency [-]",
                                   "deltaT_HTF_PHX" : "PHX Temp\nDifference [C]",
                                   "P_comp_out" : "Comp Outlet\nPressure [MPa]",
                                   "P_cooler_in" : "Turbine Outlet\nPressure [MPa]",
                                   "P_comp_in" : "Comp Inlet\nPressure [MPa]",
                                   "recomp_frac" : "Recompression\nFraction [-]"}
        
        if(cycle_config == "partialcooling"):
            cycle_name = "Partial cooling cycle"
        elif(cycle_config == "recompression"):
            cycle_name = "Recompression cycle"
        else:
            cycle_name = "Cycle"
        
        plt_title = cycle_name + " solved metrics vs recuperator conductance"
        
        save_title = cycle_config + "_vs_" + x_var + "__" + data_desc_1 + "__comp__" + data_desc_2
        
        super().__init__(cycle_config, data_1, data_2, x_var, x_label, y_var_dict_ordered, 
             plt_title, save_title,
             data_desc_1, data_desc_2)
    
class C_UA_par__stacked_outputs_comp_plot_from_json(C_UA_par__stacked_outputs_comp_plot):

    def __init__(self, cycle_config, file_end_1, file_end_2, file_1_desc = "", file_2_desc = ""):
        
        data_1 = json.load(open(cycle_config + "_" + file_end_1 + ".txt"))
        if(file_1_desc == ""):
            file_1_desc = file_end_1
            
        data_2 = json.load(open(cycle_config + "_" + file_end_2 + ".txt"))
        if(file_2_desc == ""):
            file_2_desc = file_end_2
        
        super().__init__(cycle_config, data_1, data_2, file_1_desc, file_2_desc)


def plot_eta_vs_deltaT__constant_UA__multi_configs(list_des_results, UA_fixed):
    
    plt.figure(num = 1,figsize=(7.0,3.5))
    
    plot_colors = ['k','b','g']
    
    lss = ["-","--"]
    
    mss = ["o","d","s"]
    
    for i_cycle, val_unused in enumerate(list_des_results):

        index_UA_plot = []
        
        ls_i = plot_colors[i_cycle%len(plot_colors)] + lss[i_cycle%len(lss)] + mss[i_cycle%len(mss)]
        
        cycle_config_code = list_des_results[i_cycle]["cycle_config"][0]
        if(cycle_config_code == 1):
            cycle_config = "recompression"
        elif(cycle_config_code == 2):
            cycle_config = "partialcooling"
        else:
            cycle_config = "cycle"
        print("i_cycle = ", i_cycle, " cycle config = ", cycle_config)
        
        max_eta = max(list_des_results[i_cycle]["eta_thermal_calc"])
        
        for i, val in enumerate(list_des_results[i_cycle]["UA_recup_tot_des"]):
            if val == UA_fixed:
                if (list_des_results[i_cycle]["eta_thermal_calc"][i] > 0.8*max_eta):
                    index_UA_plot.append(i)
                
        plot_data_UA = dict((key, [val[i] for i in index_UA_plot]) for key, val in list_des_results[i_cycle].items())
        
        plt.plot(plot_data_UA["deltaT_HTF_PHX"], plot_data_UA["eta_thermal_calc"], ls_i,label = cycle_config)

        print(plot_data_UA["UA_HTR"])
        print(plot_data_UA["UA_LTR"])
        for abc in range(len(plot_data_UA["UA_HTR"])-1):
            print(plot_data_UA["UA_HTR"][abc]/(plot_data_UA["UA_LTR"][abc]+plot_data_UA["UA_HTR"][abc]))
    
    plt.xlabel("PHX Temperature Difference [C]")
    
    plt.legend(numpoints = 1, markerscale = 0.7)
    
    plt.ylabel("Thermal Efficiency [-]") 
    plt.ylim(ymax = 0.5, ymin = 0.4)
    plt.yticks(np.arange(0.4, 0.501, 0.01))
    
    plt.grid(which = 'both', color = 'gray', alpha = 1) 
    plt.tight_layout(rect=(0.01,0.01,1.0,1.0))
    plt.savefig('results/' + "combined_eta_vs_deltaT.png")
    plt.close()


def plot_eta_vs_UA__add_UA_saturation_point__multi_configs(list_des_results):
    
    plt.figure(num = 1,figsize=(7.0,3.5))
    
    fs_s = 10;
    
    lss = ["-","--"]
    
    UA_max = 0.0
    
    for i_cycle, val_unused in enumerate(list_des_results):
        
        ls_i = lss[i_cycle%2]
    
        cycle_config_code = list_des_results[i_cycle]["cycle_config"][0]
        if(cycle_config_code == 1):
            cycle_config = "recompression"
        elif(cycle_config_code == 2):
            cycle_config = "partialcooling"
        else:
            cycle_config = "cycle"
        print("i_cycle = ", i_cycle, " cycle config = ", cycle_config)
    
        plt.plot(list_des_results[i_cycle]["UA_recup_tot_des"], list_des_results[i_cycle]["eta_thermal_calc"], ls_i, label = cycle_config)
        
        #max_eta = max(list_des_results[i_cycle]["eta_thermal_calc"])
        
        list_a, list_b = calculate_UA_saturated([list_des_results[i_cycle]], 0.0025/2, 0.05)
        list_des_results[i_cycle]["UA_practical"] = list_a[0]
        list_des_results[i_cycle]["eta_UA_practical"]= list_b[0]
        #list_des_results[i_cycle]["UA_practical"][0], list_des_results[i_cycle]["eta_UA_practical"][0] = calculate_UA_saturated(list_des_results[i_cycle], 0.0025/2, 0.05)
        
        # This tries to calculate the point where adding recuperator UA is "not worth it"
#        for i, val_unused_2 in enumerate(list_des_results[i_cycle]["min_phx_deltaT"]):
#            
#            if(i < len(list_des_results[i_cycle]["min_phx_deltaT"]) - 1):
#                
#                # Relative increase in UA between items
#                UA_frac = list_des_results[i_cycle]["UA_recup_tot_des"][i+1] / list_des_results[i_cycle]["UA_recup_tot_des"][i]
#                # Absolute change in efficiency between items
#                eta_increase = list_des_results[i_cycle]["eta_thermal_calc"][i+1] - list_des_results[i_cycle]["eta_thermal_calc"][i]
#                
#                # If the quotient of is equivalent to less than 0.0025 efficiency points per doubling the UA, then it's "saturated"
#                # Could also compare recuperator solution (e.g. is effectiveness == max or is UA_allocated > UA_solved)
#                if( eta_increase / UA_frac < 0.0025 /2 and list_des_results[i_cycle]["eta_thermal_calc"][i+1] > max_eta - 0.05):
#                    print(list_des_results[i_cycle]["UA_recup_tot_des"][i+1], list_des_results[i_cycle]["eta_thermal_calc"][i+1], max_eta)
#                    list_des_results[i_cycle]["UA_practical"] = list_des_results[i_cycle]["UA_recup_tot_des"][i+1]
#                    list_des_results[i_cycle]["eta_UA_practical"] = list_des_results[i_cycle]["eta_thermal_calc"][i+1]
#                    break;
#                    
#            else:
#                print ("did not find break point")
            
        # Overlay this practical maximum on eta-UA plot
        plt.plot(list_des_results[i_cycle]["UA_practical"], list_des_results[i_cycle]["eta_UA_practical"],'o')
        
        UA_max = max(UA_max, list_des_results[i_cycle]["UA_practical"])
        
    plt.ylabel("Thermal Efficiency [-]")
    plt.xlabel("Recuperator Conductance [kW/K]")
    plt.grid(which = 'both', color = 'gray', alpha = 0.75)
    plt.tight_layout(rect=(0.01,0.01,1.0,1.0))
    plt.legend(title = "Cycle Configuration", fontsize=fs_s, loc = 'lower right', labelspacing=0.25,numpoints=1)       
    plt.savefig('results/' + "combined_eta_vs_UA.png")
    plt.close()
    
    #return UA_max

def calculate_UA_saturated(list_des_results, delta_eta_abs_over_UA_frac_rel_min, eta_max_diff_floor):

    # This tries to calculate the point where adding recuperator UA is "not worth it"
    UA_max = [0]
    eta_at_UA_max = [0]

    for i_cycle in range(len(list_des_results)):        

        max_eta = max(list_des_results[i_cycle]["eta_thermal_calc"])
        
        for i, val_unused_2 in enumerate(list_des_results[i_cycle]["min_phx_deltaT"]):
            
            if(i < len(list_des_results[i_cycle]["min_phx_deltaT"]) - 1):
                
                # Relative increase in UA between items
                UA_frac = list_des_results[i_cycle]["UA_recup_tot_des"][i+1] / list_des_results[i_cycle]["UA_recup_tot_des"][i]
                # Absolute change in efficiency between items
                eta_increase = list_des_results[i_cycle]["eta_thermal_calc"][i+1] - list_des_results[i_cycle]["eta_thermal_calc"][i]
                
                # If the quotient of is equivalent to less than 0.0025 efficiency points per doubling the UA, then it's "saturated"
                # Could also compare recuperator solution (e.g. is effectiveness == max or is UA_allocated > UA_solved)
                #if( eta_increase / UA_frac < 0.0025 /2 and list_des_results[i_cycle]["eta_thermal_calc"][i+1] > max_eta - 0.05)
                if( eta_increase / UA_frac < delta_eta_abs_over_UA_frac_rel_min and list_des_results[i_cycle]["eta_thermal_calc"][i+1] > max_eta - eta_max_diff_floor):
                    print(list_des_results[i_cycle]["UA_recup_tot_des"][i+1], list_des_results[i_cycle]["eta_thermal_calc"][i+1], max_eta)
                    UA_practical = list_des_results[i_cycle]["UA_recup_tot_des"][i+1]
                    eta_practical = list_des_results[i_cycle]["eta_thermal_calc"][i+1]
                    break;
                    
            else:
                UA_practical = list_des_results[i_cycle]["UA_recup_tot_des"][i]
                eta_practical = list_des_results[i_cycle]["eta_thermal_calc"][i]
                
        if(i_cycle==0):
            UA_max[0] = UA_practical
            eta_at_UA_max[0] = eta_practical
        else:
            UA_max.append(UA_practical)
            eta_at_UA_max.append(eta_practical)
    
    return UA_max, eta_at_UA_max

def plot_eta_vs_UA__deltaT_levels__two_config(list_des_results):
    
    plot_ls = ["-","--"]
    color_list = ['k','b','r','g','y','m','c','k','b','r','g']
    
    color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    plot_keys = ["min_phx_deltaT", "UA_recup_tot_des", "eta_thermal_calc", "deltaT_HTF_PHX"]
    
    legend_lines = []
    
    if True:
        x_var = "UA_recup_tot_des"
        x_label = "Total Recuperator Conductance [kW/K]"
        y_var = "eta_thermal_calc"
        y_label = "Thermal Efficiency [-]"
        overlay_key = "min_phx_deltaT"
        overlay_label = " [C]"
        overlay_title = "Minimum PHX\nTemp Difference"
        save_title = "_eta_vs_UA_PHX_dT_par.png"
        
    elif False:
        x_var = "deltaT_HTF_PHX"
        x_label = "PHX Temperature Difference [C]"
        y_var = "eta_thermal_calc"
        y_label = "Thermal Efficiency [-]"
        overlay_key = "min_phx_deltaT"
        overlay_label = " [C]"
        overlay_title = "Minimum PHX\nTemp Difference"
        save_title = "_eta_vs_real_phx_dT.png"
        
    elif True:
        x_var = "deltaT_HTF_PHX"
        x_label = "PHX Temperature Difference [C]"
        y_var = "eta_thermal_calc"
        y_label = "Thermal Efficiency [-]"
        overlay_key = "UA_recup_tot_des"
        overlay_label = " [kW/K]"
        overlay_title = "Total Recup\n Conductance"
        save_title = "_eta_vs_real_phx_dT_UA.png"
    
    plt.figure(num = 1,figsize=(7.0,3.5))
    
    style_key = {}
    
    n_cycles = len(list_des_results)
    
    #for i_cycle, cycle_config in enumerate(cycle_configs):
    for i_cycle, val_unused in enumerate(list_des_results):
    
        cycle_config_code = list_des_results[i_cycle]["cycle_config"][0]
        if(cycle_config_code == 1):
            cycle_config = "recompression"
        elif(cycle_config_code == 2):
            cycle_config = "partialcooling"
        else:
            cycle_config = "cycle"
        print("i_cycle = ", i_cycle, " cycle config = ", cycle_config)
        
        legend_lines.append(mlines.Line2D([],[],color='k', ls = plot_ls[i_cycle], label = cycle_config))
        
        # Load results of min_deltaT -> UA parametric
        #data = json.load(open(cycle_config + "_dT_UA_par_sweep_q2_baseline.txt"))
        data = list_des_results[i_cycle]
        
        # Filter dictionary to contain only keys that we want for plotting
        plot_data_filtered = {k:v for (k,v) in data.items() if k in plot_keys}
        
        # Sort remaining data by outer next variable: min_phx_deltaT
        mdT = data[overlay_key]
        
        unique_mdTs = sorted(list(set(mdT)), reverse=False)
        
        plot_data = {}
        
        for dT in unique_mdTs:
    
            dT_key = str(dT)
            plot_data[dT_key] = {}
            
            plot_data[dT_key]["dT"] = dT
            
            plot_data_sorted_ind_mdTs = []
            
            for i, val in enumerate(mdT):
                if val == dT:
                    plot_data_sorted_ind_mdTs.append(i)
            
            plot_data[dT_key] = dict((key, [val[i] for i in plot_data_sorted_ind_mdTs]) for key, val in plot_data_filtered.items())
    
        if(i_cycle == 0):
            style_key = plot_data
    
        fs_s = 10;
        
        for i,key in enumerate(plot_data):
            #print(plot_data[key][x_var])
            
            if(i_cycle == 0):
                co_i = i
            else:
                co_i = i
                for k,k_key in enumerate(style_key):
                    if(k_key == key):
                        co_i = k
                        
            if(i_cycle == 0):
                plt.plot(plot_data[key][x_var],plot_data[key][y_var], plot_ls[i_cycle], color = color_list[co_i], markersize = 5, lw = 2, label = key + overlay_label)
            else:
                plt.plot(plot_data[key][x_var],plot_data[key][y_var], plot_ls[i_cycle], color = color_list[co_i], markersize = 5, lw = 2)
           
    legend1 = plt.legend(handles=legend_lines, fontsize = 8, ncol = 2, loc = "center", bbox_to_anchor = (0.5,1.05))
    plt.legend(title = overlay_title, fontsize=fs_s, loc = 'upper left', bbox_to_anchor = (1.0,1.0), labelspacing=0.25,numpoints=1)
    
    plt.gca().add_artist(legend1)
    
    plt.tight_layout(rect=(0.02,0.02,0.8,0.96))
    
    plt.ylim(ymax = 0.5, ymin = 0.3)
    plt.yticks(np.arange(0.3, 0.501, 0.02))
    plt.ylabel(y_label)
    plt.grid(which = 'both', color = 'gray', alpha = 0.5) 
    
    plt.xlim(xmax = 50000)
    plt.xlabel(x_label)   
    
    if(n_cycles == 1):
        plt.savefig('results/' + cycle_config + save_title)
    else:
        plt.savefig('results/' + "overlay_cycles" + save_title)
        
    plt.close()

class C_plot_udpc_results:

    class C_settings:

        def __init__(self):
            self.plot_pre_str = ""
            self.cycle_des_str = ""
            self.is_T_t_in_set = False
            self.is_six_plots = False
            self.udpc_check_dict = ""
            self.is_plot_regression = True
            self.is_plot_interp = True
            self.LT_udpc_table_m_dot_sweep = ""
            self.HT_udpc_table_m_dot_sweep = ""
            self.is_three_plots = True

    class udpc_col_and_label_struct:

        def __init__(self, col, label):
            self.col = col
            self.label = label

    class UDPC_COLS(Enum):
        T_HTF = 0
        M_DOT_ND = 1
        T_AMB = 2
        W_GROSS_ND = 3
        Q_ND = 4
        W_PAR_ND = 5
        M_DOT_WATER_ND = 6
        DELTA_T_ND = 7
        T_CO2_PHX_IN_ND = 8
        M_DOT_T_ND = 9
        P_T_IN_ND = 10
        W_NET_ND = 11
        ETA_NET_ND = 12
        T_HTF_COLD_ND = 13

    class udpc_cols_and_labels:

        def __init__(self):
            
            self.T_HTF = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.T_HTF.value, "HTF Hot Temperature [C]")
            self.M_DOT_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.M_DOT_ND.value, "Normalized HTF Mass Flow")
            self.T_AMB = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.T_AMB.value, "Ambient Temperature [C]")
            self.W_GROSS_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.W_GROSS_ND.value, "Normalized Gross Power")
            self.Q_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.Q_ND.value, "Normalized Heat Input")
            self.W_PAR_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.W_PAR_ND.value, "Normalized Parasitics")
            self.M_DOT_WATER_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.M_DOT_WATER_ND.value, "Normalized Water Use")
            self.DELTA_T_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.DELTA_T_ND.value, "Normalized HTF Temp Diff")
            self.T_CO2_PHX_IN_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.T_CO2_PHX_IN_ND.value, "Normalized Turb In Temp")
            self.M_DOT_T_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.M_DOT_T_ND.value, "Normalized Turb Mass Flow")
            self.P_T_IN_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.P_T_IN_ND.value, "Normalized Turb In Pres")
            self.W_NET_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.W_NET_ND.value, "Normalized Net Power")
            self.ETA_NET_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.ETA_NET_ND.value, "Normalized Net Efficiency")
            self.T_HTF_COLD_ND = C_plot_udpc_results.udpc_col_and_label_struct(C_plot_udpc_results.UDPC_COLS.T_HTF_COLD_ND.value, "Normalized HTF Cold Temp")

    def __init__(self, udpc_data, n_T_htf, n_T_amb, n_m_dot_htf, settings):
        self.udpc_data_base = udpc_data
        self.n_T_htf = n_T_htf
        self.n_T_amb = n_T_amb
        self.n_m_dot_htf = n_m_dot_htf
        self.settings = settings
        self.cols_and_labels = C_plot_udpc_results.udpc_cols_and_labels()

    def update_settings(self, settings):
        self.settings = settings

    def make_udpc_plots(self):

        # Make copy here, because we're adding two columns below
        # and we don't want to modify the class member udpc data
        udpc_data = copy.deepcopy(self.udpc_data_base)

        is_plot_tests = False
        if self.settings.udpc_check_dict != "":
            if self.settings.udpc_check_dict["plot_udpc_tests"]:
                is_plot_tests = True

        n_levels = 3
        l_color = ['k', 'b', 'r']
        ls_basis = "-"
        pt_mrk = "o"
        s_subplot = ["a", "b", "c", "d", "e", "f"]

        

        f_udpc_pars = open(self.settings.plot_pre_str + "_udpc_setup_pars.txt", 'w')
        f_udpc_pars.write(self.settings.cycle_des_str)

        if(self.settings.is_T_t_in_set):
            f_udpc_pars.write("Number of turbine inlet temperature levels = " + str(self.n_T_htf) + "\n")
            f_udpc_pars.write("Number of target output power levels = " + str(self.n_m_dot_htf) + "\n")
            m_dot_str = "target_power_ND = "
        else:
            f_udpc_pars.write("Number of HTF hot temperature levels = " + str(self.n_T_htf) + "\n")
            f_udpc_pars.write("Number of HTF mass flow rate levels = " + str(self.n_m_dot_htf) + "\n")
            m_dot_str = "m_dot_ND = "

        f_udpc_pars.write("Number of ambient temperature levels = " + str(self.n_T_amb) + "\n")

        # UDPC column definition from sco2_csp_system
        # See class UDPC_COLS(Enum)
        # 0) HTF Temp [C], 1) HTF ND mass flow [-], 2) Ambient Temp [C], 3) ND "gross" Power, 4) ND Heat In, 5) ND Fan Power, 6) ND Water
        # 7) deltaT_ND, 8) P_co2_OHX_in, 9) t_m_dot, 10) t_P_in
        # 11) ND W_dot_net 12) ND eta_net, 13) T_htf_cold_diff / deltaT_des

        len_udpc_base = len(udpc_data[0])
        #print("udpc row length = ", len_udpc_base)

        # Add normalized efficiency column
        for row in udpc_data:
            row.append(row[self.UDPC_COLS.W_GROSS_ND.value] / row[self.UDPC_COLS.Q_ND.value])  # Adding *normalized* gross cycle and parasitic values, which doesn't really make sense
            ADD_COL_ETA_GROSS = C_plot_udpc_results.udpc_col_and_label_struct(len_udpc_base, "Normalized Gross Efficiency")
            
        # Choose variables to plot
        if(len_udpc_base == 14):
            
            if self.settings.is_three_plots:

                f_h = 4
                w_pad = 1

                mi = [[0, 0, self.cols_and_labels.W_NET_ND]]
                mi.append([0, 1, self.cols_and_labels.ETA_NET_ND])
                mi.append([0, 2, self.cols_and_labels.Q_ND])
                nrows = 1
                ncols = 3
            else:       

                f_h = 10/3.*nrows
                w_pad = 3

                mi = [[0, 0, self.cols_and_labels.W_NET_ND]]
                mi.append([1, 0, self.cols_and_labels.ETA_NET_ND])
                mi.append([0, 1, self.cols_and_labels.Q_ND])
                if(self.settings.is_T_t_in_set):
                    mi.append([1, 1, self.cols_and_labels.DELTA_T_ND])
                else:
                    mi.append([1, 1, self.cols_and_labels.W_PAR_ND])
                    #mi.append([1, 1, self.cols_and_labels.DELTA_T_ND])
                    #mi.append([1, 1, ADD_COL_ETA_GROSS])
                nrows = 2
                ncols = 2
            
                if(self.settings.is_six_plots):
                    mi.append([2, 0, self.cols_and_labels.W_PAR_ND])
                    mi.append([2, 1, self.cols_and_labels.W_GROSS_ND])
                    nrows = 3

        # 22-12-29: Leaving this for backwards compatibility with sco2 code, but untested
        else:
            print("The updc input data does not have 14 columns, so it was likely generated with legacy SSC code."
            " We recommend using the latest SSC sco2 code to generate results. If you choose to continue with this data"
            " please review the generated plots and files to ensure they properly read and plot your data")
            mi = [[0, 0, self.cols_and_labels.W_GROSS_ND]]
            mi.append([1, 0, ADD_COL_ETA_GROSS])
            mi.append([0, 1, self.cols_and_labels.Q_ND])
            if(self.settings.is_T_t_in_set):
                mi.append([1, 1, self.cols_and_labels.DELTA_T_ND])
            else:
                mi.append([1, 1, self.cols_and_labels.DELTA_T_ND])
            nrows = 2
            ncols = 2
        
            if(self.settings.is_six_plots):
                mi.append([2, 0, self.cols_and_labels.W_PAR_ND])
                mi.append([2, 1, self.cols_and_labels.M_DOT_T_ND])
                nrows = 3
        ###################################################################################
        ###################################################################################

        
        fig1, a_ax = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(7.48, f_h))

        # T_htf parametric values, 3 m_dot levels, design ambient temperature
        for j in range(0, len(mi)):
            
            if nrows > 1:
                j_ax = a_ax[mi[j][0], mi[j][1]]
            else:
                j_ax = a_ax[mi[j][1]]

            for i in range(0, n_levels):
                row_start = i * self.n_T_htf
                row_end = i * self.n_T_htf + self.n_T_htf
                if( j == 0 ):
                    j_ax.plot([k[0] for k in udpc_data[row_start:row_end]],
                        [k[mi[j][2].col] for k in udpc_data[row_start:row_end]],l_color[i]+ls_basis+pt_mrk,
                            label = m_dot_str + str(udpc_data[row_start][1]), markersize = 2.4)
                    if(i == 0):
                        f_udpc_pars.write("Mass flow rate Low Level = " + str(udpc_data[row_start][1]) + "\n")
                    if(i == 1):
                        f_udpc_pars.write("Mass flow rate Design Level = " + str(udpc_data[row_start][1]) + "\n")
                    if(i == 2):
                        f_udpc_pars.write("Mass flow rate High Level = " + str(udpc_data[row_start][1]) + "\n")
                else:
                    j_ax.plot([k[0] for k in udpc_data[row_start:row_end]],
                            [k[mi[j][2].col] for k in udpc_data[row_start:row_end]],l_color[i]+ls_basis+pt_mrk, markersize = 2.4)
            if (self.settings.is_T_t_in_set):
                j_ax.set_xlabel("Turbine Inlet Temperature [C]")
            else:
                j_ax.set_xlabel("HTF Hot Temperature [C]")
            j_ax.set_ylabel(mi[j][2].label)
            j_ax.grid(which='both', color='gray', alpha=1)

        top_layout_mult = 0.94
        right_layout_mult = 0.98
        fig1.legend(ncol=n_levels, loc="upper center", columnspacing=0.6, bbox_to_anchor=(0.5, 1.0))
        plt.tight_layout(pad=0.0, h_pad=1, w_pad=w_pad, rect=(0.012, 0.02, right_layout_mult, top_layout_mult))
        plt.savefig('results/' + self.settings.plot_pre_str + "_udpc_T_HTF.png")
        plt.close()

        fig1, a_ax = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(7.48, f_h))

        if is_plot_tests:
            T_amb_LT = self.settings.udpc_check_dict["T_amb_LT"]
            T_amb_HT = self.settings.udpc_check_dict["T_amb_HT"]
            color_HT = 'g'
            color_LT = 'm'

            ls_interp = ":"
            mrk_interp = 's'

            if self.settings.is_plot_regression:
                ls_regr = "--"
                mrk_regr = '^'

        # T_amb parametric values, 3 T_HTF_levels, design m_dot
        for j in range(0, len(mi)):

            if nrows > 1:
                j_ax = a_ax[mi[j][0], mi[j][1]]
            else:
                j_ax = a_ax[mi[j][1]]

            # Check if design and upper levels are very close
            is_skip_high = False
            diff_high_to_des = udpc_data[3*self.n_T_htf + 2*self.n_T_amb][0] - udpc_data[3*self.n_T_htf + self.n_T_amb][0]
            if(diff_high_to_des <= 1):
                is_skip_high = True

            for i in range(0, n_levels):
                row_start = 3 * self.n_T_htf + i * self.n_T_amb
                row_end = row_start + self.n_T_amb

                # if skip high level then don't plot but make sure to advance row start and end counters
                if(not(is_skip_high and i == 2)):

                    udpc_col_y_data = mi[j][2].col
                    y_data = [k[udpc_col_y_data] for k in udpc_data[row_start:row_end]]

                    x_data = [k[2] for k in udpc_data[row_start:row_end]]

                    if( j == 0 ):
                        j_ax.plot(x_data,
                            y_data,l_color[i]+ls_basis+pt_mrk,
                                label = "T_HTF = " + str(udpc_data[row_start][0]), markersize = 2.4)
                        if (i == 0):
                            f_udpc_pars.write("HTF temperature Low Level = " + str(udpc_data[row_start][0]) + "\n")
                        if (i == 1):
                            y_j0_des = copy.deepcopy(y_data)
                            x_j0_des = copy.deepcopy(x_data)
                            f_udpc_pars.write("HTF temperature Design Level = " + str(udpc_data[row_start][0]) + "\n")
                        if (i == 2):
                            f_udpc_pars.write("HTF temperature High Level = " + str(udpc_data[row_start][0]) + "\n")
                    else:                
                        j_ax.plot(x_data,
                                y_data, l_color[i]+ls_basis+pt_mrk, markersize = 2.4)
                        
                        if j == 1:
                            y_j1_des = copy.deepcopy(y_data)
                            x_j1_des = copy.deepcopy(x_data)

                        if j == 2:
                            y_j2_des = copy.deepcopy(y_data)
                            x_j2_des = copy.deepcopy(x_data)

            if is_plot_tests:
            
                if(j == 0):

                    if self.settings.is_plot_regression:

                        j_ax.plot(self.settings.udpc_check_dict["T_amb_pars"], self.settings.udpc_check_dict["W_dot_ND_regr_vs_T_amb__T_HTF_low_level"], l_color[0]+pt_mrk+ls_regr, markersize = 2.4)
                        
                if(j == 1):

                    if self.settings.is_plot_regression:

                        j_ax.plot(self.settings.udpc_check_dict["T_amb_pars"], self.settings.udpc_check_dict["eta_ND_regr_vs_T_amb__T_HTF_low_level"], l_color[0]+pt_mrk+ls_regr, markersize = 2.4)
                        
                if(j == 2):

                    if self.settings.is_plot_regression:

                        j_ax.plot(self.settings.udpc_check_dict["T_amb_pars"], self.settings.udpc_check_dict["q_dot_ND_regr_vs_T_amb__T_HTF_low_level"], l_color[0]+pt_mrk+ls_regr, markersize = 2.4)
                        

            j_ax.set_xlabel("Ambient Temperature [C]")
            j_ax.set_ylabel(mi[j][2].label)
            j_ax.grid(which='both', color='gray', alpha=1)

        fig1.legend(ncol=n_levels, loc="upper center", columnspacing=0.6, bbox_to_anchor=(0.5, 1.0))
        plt.tight_layout(pad=0.0, h_pad=1, w_pad=w_pad, rect=(0.012, 0.02, right_layout_mult, top_layout_mult))
        plt.savefig('results/' + self.settings.plot_pre_str + "_udpc_T_amb.png")
        plt.close()

        fig2, a_ax = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(7.48, f_h))

        # m_dot parametric values, 3 T_amb levels, design T_htf_hot
        T_low_level = -999
        T_amb_des = -999
        T_high_level = -999
        for j in range(0, len(mi)):
           
            if nrows > 1:
                j_ax = a_ax[mi[j][0], mi[j][1]]
            else:
                j_ax = a_ax[mi[j][1]]

            for i in range(0, n_levels):
                row_start = 3 * self.n_T_htf + 3 * self.n_T_amb + i * self.n_m_dot_htf
                row_end = row_start + self.n_m_dot_htf

                udpc_col_y_data = mi[j][2].col
                y_data = [k[udpc_col_y_data] for k in udpc_data[row_start:row_end]]

                if(udpc_col_y_data == self.cols_and_labels.ETA_NET_ND.col and i == 2 and False):
                    print("i = ", i, " y data = ", y_data)
                    list_line_props(y_data, 0.025)

                j_ax.plot([k[self.cols_and_labels.M_DOT_ND.col] for k in udpc_data[row_start:row_end]],
                        y_data,l_color[i]+ls_basis+pt_mrk, markersize = 2.4) # label = "T_amb = " + str(udpc_data[row_start][2]),

                if( j == 0 ):
                     
                    if (i == 0):
                        T_low_level = udpc_data[row_start][self.cols_and_labels.T_AMB.col]
                        f_udpc_pars.write("Ambient temperature Low Level = " + str(T_low_level) + "\n")
                    if (i == 1):
                        T_amb_des = udpc_data[row_start][self.cols_and_labels.T_AMB.col]
                        f_udpc_pars.write("Ambient temperature Design Level = " + str(T_amb_des) + "\n")
                    if (i == 2):
                        T_high_level = udpc_data[row_start][self.cols_and_labels.T_AMB.col]
                        f_udpc_pars.write("Ambient temperature High Level = " + str(T_high_level) + "\n")
         
            if is_plot_tests:
                
                if(j == 0):

                    # Basis model at LT
                    if(self.settings.LT_udpc_table_m_dot_sweep != ""):
                        j_ax.plot([k[1] for k in self.settings.LT_udpc_table_m_dot_sweep],
                        [k[mi[j][2].col] for k in self.settings.LT_udpc_table_m_dot_sweep], color_LT+pt_mrk+ls_basis, markersize = 2.4)

                    # Basis model at HT
                    if(self.settings.HT_udpc_table_m_dot_sweep != ""):
                        j_ax.plot([k[1] for k in self.settings.HT_udpc_table_m_dot_sweep],
                        [k[mi[j][2].col] for k in self.settings.HT_udpc_table_m_dot_sweep], color_HT+pt_mrk+ls_basis, markersize = 2.4)

                    if self.settings.is_plot_interp:
                        # UDPC interpolated max points
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_low_level_rule0"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_low_level_rule0"], l_color[0]+mrk_interp)                   
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_design_rule0"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_design_rule0"], l_color[1]+mrk_interp)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_high_level_rule0"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_high_level_rule0"], l_color[2]+mrk_interp)                     
                        
                        # UDPC interpolated HT curve
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["W_dot_ND_vs_m_dot__T_amb_HT"], color_HT+pt_mrk+ls_interp, markersize = 2.4)  
                        # UDPC interpolated HT max point
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_HT_rule0"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_HT_rule0"], color_HT+mrk_interp)

                        # UDPC interpolated LT curve
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["W_dot_ND_vs_m_dot__T_amb_LT"], color_LT+pt_mrk+ls_interp, markersize = 2.4)
                        # UDPC interpolated LT max point
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_LT_rule0"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_LT_rule0"], color_LT+mrk_interp)

                    if self.settings.is_plot_regression:
                        # UDPC regression  curves    
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["W_dot_ND_regr_vs_m_dot__T_amb_high_level"], l_color[2]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_high_level_regr"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_high_level_regr"], l_color[2]+mrk_regr)
                                            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["W_dot_ND_regr_vs_m_dot__T_amb_design"], l_color[1]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_design_regr"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_design_regr"], l_color[1]+mrk_regr)
                        
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["W_dot_ND_regr_vs_m_dot__T_amb_low_level"], l_color[0]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_low_level_regr"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_low_level_regr"], l_color[0]+mrk_regr)
                                            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["W_dot_ND_regr_vs_m_dot__T_amb_LT"], color_LT+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_LT_regr"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_LT_regr"], color_LT+mrk_regr)                               
                        
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["W_dot_ND_regr_vs_m_dot__T_amb_HT"], color_HT+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_HT_regr"], self.settings.udpc_check_dict["W_dot_htf_ND_max_at_T_amb_HT_regr"], color_HT+mrk_regr)


                if(j == 1):
                    
                    # Basis model at LT
                    if(self.settings.LT_udpc_table_m_dot_sweep != ""):
                        j_ax.plot([k[1] for k in self.settings.LT_udpc_table_m_dot_sweep],
                        [k[mi[j][2].col] for k in self.settings.LT_udpc_table_m_dot_sweep], color_LT+pt_mrk+ls_basis, markersize = 2.4)

                    # Basis model at HT
                    if(self.settings.HT_udpc_table_m_dot_sweep != ""):
                        j_ax.plot([k[1] for k in self.settings.HT_udpc_table_m_dot_sweep],
                        [k[mi[j][2].col] for k in self.settings.HT_udpc_table_m_dot_sweep], color_HT+pt_mrk+ls_basis, markersize = 2.4)

                    if self.settings.is_plot_interp:
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_low_level_rule0"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_low_level_rule0"], l_color[0]+mrk_interp)                
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_design_rule0"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_design_rule0"], l_color[1]+mrk_interp)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_high_level_rule0"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_high_level_rule0"], l_color[2]+mrk_interp)               

                        # UDPC interpolated HT curve
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["eta_ND_vs_m_dot__T_amb_HT"], color_HT+pt_mrk+ls_interp, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_HT_rule0"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_HT_rule0"], color_HT+mrk_interp)
                        
                        # UDPC interpolated LT curve
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["eta_ND_vs_m_dot__T_amb_LT"], color_LT+pt_mrk+ls_interp, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_LT_rule0"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_LT_rule0"], color_LT+mrk_interp)

                    if self.settings.is_plot_regression:
                        # UDPC regression curves                
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["eta_ND_regr_vs_m_dot__T_amb_high_level"], l_color[2]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_high_level_regr"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_high_level_regr"], l_color[2]+mrk_regr)
                            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["eta_ND_regr_vs_m_dot__T_amb_design"], l_color[1]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_design_regr"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_design_regr"], l_color[1]+mrk_regr)
                            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["eta_ND_regr_vs_m_dot__T_amb_low_level"], l_color[0]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_low_level_regr"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_low_level_regr"], l_color[0]+mrk_regr)
                            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["eta_ND_regr_vs_m_dot__T_amb_LT"], color_LT+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_LT_regr"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_LT_regr"], color_LT+mrk_regr)
                        
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["eta_ND_regr_vs_m_dot__T_amb_HT"], color_HT+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_HT_regr"], self.settings.udpc_check_dict["eta_ND_max_at_T_amb_HT_regr"], color_HT+mrk_regr)

                if(j == 2):
                    
                    # Basis model at LT
                    if(self.settings.LT_udpc_table_m_dot_sweep != ""):
                        j_ax.plot([k[1] for k in self.settings.LT_udpc_table_m_dot_sweep],
                        [k[mi[j][2].col] for k in self.settings.LT_udpc_table_m_dot_sweep], color_LT+pt_mrk+ls_basis, markersize = 2.4)

                    # Basis model at HT
                    if(self.settings.HT_udpc_table_m_dot_sweep != ""):
                        j_ax.plot([k[1] for k in self.settings.HT_udpc_table_m_dot_sweep],
                        [k[mi[j][2].col] for k in self.settings.HT_udpc_table_m_dot_sweep], color_HT+pt_mrk+ls_basis, markersize = 2.4)

                    if self.settings.is_plot_interp:
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_low_level_rule0"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_low_level_rule0"], l_color[0]+mrk_interp)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_design_rule0"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_design_rule0"], l_color[1]+mrk_interp)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_high_level_rule0"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_high_level_rule0"], l_color[2]+mrk_interp)
            
                        # UDPC interpolated HT curve
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["q_dot_ND_vs_m_dot__T_amb_HT"], color_HT+pt_mrk+ls_interp,markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_HT_rule0"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_HT_rule0"], color_HT+mrk_interp)

                        # UDPC interpolated LT curve
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["q_dot_ND_vs_m_dot__T_amb_LT"], color_LT+pt_mrk+ls_interp,markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_LT_rule0"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_LT_rule0"], color_LT+mrk_interp)
                        
                    if self.settings.is_plot_regression:
                        # UDPC regression curves
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["q_dot_ND_regr_vs_m_dot__T_amb_high_level"], l_color[2]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_high_level_regr"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_high_level_regr"], l_color[2]+mrk_regr)
            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["q_dot_ND_regr_vs_m_dot__T_amb_design"], l_color[1]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_design_regr"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_design_regr"], l_color[1]+mrk_regr)
                
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["q_dot_ND_regr_vs_m_dot__T_amb_low_level"], l_color[0]+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_low_level_regr"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_low_level_regr"], l_color[0]+mrk_regr)
            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["q_dot_ND_regr_vs_m_dot__T_amb_LT"], color_LT+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_LT_regr"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_LT_regr"], color_LT+mrk_regr)
            
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_pars"], self.settings.udpc_check_dict["q_dot_ND_regr_vs_m_dot__T_amb_HT"], color_HT+pt_mrk+ls_regr, markersize = 2.4)
                        j_ax.plot(self.settings.udpc_check_dict["m_dot_htf_ND_max_at_T_amb_HT_regr"], self.settings.udpc_check_dict["q_dot_htf_ND_max_at_T_amb_HT_regr"], color_HT+mrk_regr)
                        

            if (self.settings.is_T_t_in_set):
                j_ax.set_xlabel("Normalized Target Power Output")
            else:
                j_ax.set_xlabel("Normalized HTF Mass Flow")
            j_ax.set_ylabel(s_subplot[j] + ") " + mi[j][2].label)
            j_ax.grid(which='both', color='gray', alpha=1)

        if is_plot_tests:
            color_temp_list = [(l_color[0], T_low_level), (l_color[1], T_amb_des), (l_color[2], T_high_level), (color_LT, T_amb_LT), (color_HT, T_amb_HT)]

        else:
            color_temp_list = [(l_color[0], T_low_level), (l_color[1], T_amb_des), (l_color[2], T_high_level)]

        color_temp_list.sort(key=lambda x: x[1])
        color_patch_legend = []
        for i_ct in color_temp_list:
            color_patch_legend.append(mlines.Line2D([], [], color=i_ct[0], linestyle='-', label=str(i_ct[1])))

        y_top = 0.94

        if is_plot_tests:

            if nrows == 1:
                y_top = 0.8
            else:
                y_top = 0.98

            # Line styles
            line_model_list = [(ls_basis, "Reference")]
            if self.settings.is_plot_interp:
                line_model_list.append((ls_interp, "UDPC Interpolation"))
            if self.settings.is_plot_regression:
                line_model_list.append((ls_regr, "Engineering Heuristic"))
            lm_patch_legend = []
            for i_lm in line_model_list:
                lm_patch_legend.append(mlines.Line2D([], [], color='k', linestyle=i_lm[0], label=i_lm[1]))

            # Marker styles
            marker_max_list = []
            if self.settings.is_plot_interp:
                marker_max_list.append((mrk_interp, "UDPC Interpolation"))
            if self.settings.is_plot_regression:
                marker_max_list.append((mrk_regr, "Engineering Heuristic"))
            mm_patch_legend = []
            for i_mm in marker_max_list:
                mm_patch_legend.append(mlines.Line2D([], [], color='k', marker=i_mm[0], label=i_mm[1]))

            fig2.legend(handles=lm_patch_legend, loc="upper right", fontsize = 8, title = "Model Type") #, bbox_to_anchor = (0.5,1.0))
            if len(mm_patch_legend) > 0:
                fig2.legend(handles=mm_patch_legend, loc="upper center", fontsize = 8, title = "Max Operating Point") #, bbox_to_anchor = (0.5,1.0))
            fig2.legend(handles=color_patch_legend, ncol = n_levels,  loc="upper left", columnspacing = 0.6, fontsize = 8, title = "Ambient Temperature [C]") #, bbox_to_anchor = (0.5,1.0))


        else:
            
            #fig2.legend(ncol = n_levels, loc = "upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0))
            fig2.legend(handles=color_patch_legend, ncol = n_levels, loc="upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0), title = "Ambient Temperature [C]") #, bbox_to_anchor = (0.5,1.0))
        
        plt.tight_layout(pad=0.0, h_pad=1, w_pad=w_pad, rect=(0.012, 0.02, right_layout_mult, y_top))

        plt.savefig('results/' + self.settings.plot_pre_str + "_udpc_m_dot_htf.png")
        plt.close()

        return x_j0_des, y_j0_des, x_j1_des, y_j1_des, x_j2_des, y_j2_des
    
def list_line_props(list_in, const_delta_x):

    null = float('nan')
    i_prev = null
    i_0 = null
    i_next = null

    slope_ahead_prev = null
    slope_ahead_0 = null

    delta_x = const_delta_x

    if(len(list_in) < 2):
        print("need a list length greater than 2")
        return

    l_slope_ahead = []
    l_slope_slope = []

    for i in range(len(list_in)-1):
        
        i_prev = i_0
        i_0 = list_in[i]     
        i_next = list_in[i+1]

        slope_ahead_prev = slope_ahead_0

        slope_ahead_0 = (i_next - i_0) / delta_x

        slope_slope = (slope_ahead_0 - slope_ahead_prev) / delta_x

        l_slope_ahead.append(slope_ahead_0)
        l_slope_slope.append(slope_slope)

    print("slope ahead = ", l_slope_ahead)
    print("slope slope = ", l_slope_slope)


def plot_udpc_results(udpc_data, n_T_htf, n_T_amb, n_m_dot_htf, plot_pre_str = "", cycle_des_str = "", 
                        is_T_t_in_set = False, is_six_plots = False, udpc_check_dict = ""):

    
    plot_settings = C_plot_udpc_results.C_settings()
    plot_settings.plot_pre_str = plot_pre_str
    plot_settings.cycle_des_str = cycle_des_str
    plot_settings.is_T_t_in_set = is_T_t_in_set
    plot_settings.is_six_plots =  is_six_plots
    plot_settings.udpc_check_dict = udpc_check_dict
        
    c_plot_udpc = C_plot_udpc_results(udpc_data, n_T_htf, n_T_amb, n_m_dot_htf, plot_settings)

    c_plot_udpc.make_udpc_plots()

def plot_compare_udpc_results(list_of_udpc_data, n_T_htf, n_T_amb, n_m_dot_htf, plot_pre_str = "", is_six_plots = False):

    n_udpcs = len(list_of_udpc_data)
    print("Number of UPDC data sets = ",  n_udpcs)

    if(n_udpcs > 2):
        print("Can only enter two UDCPs")
        return

    n_levels = 3
    lcolor = ["k", "b", "r"]
    lstyle = ["-", "--"]

    w_pad = 3

    nrows = 2
    ncols = 2
    if(is_six_plots):
        nrows = 3

    f_h = 10/3.*nrows
    fig0, a_ax0 = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(7, f_h))


    for i_udpc, local_udpc_data in enumerate(list_of_udpc_data):

        # Add normalized efficiency column
        for row in local_udpc_data:
            row.append(row[4] / row[1])
            row.append(row[3] / row[4])
            
        # Choose variables to plot
        # mi: [x subplot 0 left, y subplot 0 top, UDPC column, label]
        mi = [[0, 0, 3, "Normalized Power"]]
        mi.append([1, 0, len(local_udpc_data[0])-1, "Normalized Efficiency"])
        #mi.append([0, 1, 5, "Normalized Cooling Power"])
        mi.append([0, 1, 4, "Normalized Heat Input"])
        mi.append([1, 1, 7, "Normalized PHX HTF deltaT"])
        
        
        if(is_six_plots):
            #mi.append(([1, 1, 7, "Normalized PHX deltaT"]))
            mi.append([2, 0, 8, "Normalized PHX Inlet Pressure"])
            mi.append([2, 1, 9, "Normalized PHX CO2 Mass Flow"])
            nrows = 3

        
        # T_htf parametric values, 3 m_dot levels, design ambient temperature
        for j in range(0, len(mi)):
            j_ax = a_ax0[mi[j][0], mi[j][1]]
            for i in range(0, n_levels):
                row_start = i * n_T_htf
                row_end = i * n_T_htf + n_T_htf
 
                if( j == 0 ):
                    j_ax.plot([k[0] for k in local_udpc_data[row_start:row_end]],
                        [k[mi[j][2]] for k in local_udpc_data[row_start:row_end]],lcolor[i]+lstyle[i_udpc],
                            label = "m_dot_ND, Case " + str(i_udpc+1) + " = " + str(local_udpc_data[row_start][1]))

                else:
                    j_ax.plot([k[0] for k in local_udpc_data[row_start:row_end]],
                        [k[mi[j][2]] for k in local_udpc_data[row_start:row_end]],lcolor[i]+lstyle[i_udpc])

            if(i_udpc == 0):
                j_ax.set_xlabel("HTF Hot Temperature [C]")
                j_ax.set_ylabel(mi[j][3])
                j_ax.grid(which='both', color='gray', alpha=1)

    fig0.legend(ncol=n_udpcs, loc="upper center", columnspacing=0.6, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(pad=0.0, h_pad=1, w_pad=w_pad, rect=(0.012, 0.02, 0.98, 0.86))
    plt.savefig('results/' + plot_pre_str + "udpc_comp_T_HTF.png")
    plt.close()

    

    fig01, a_ax01 = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(7, f_h))

    for i_udpc, local_udpc_data in enumerate(list_of_udpc_data):

        # T_amb parametric values, 3 T_HTF_levels, design m_dot
        for j in range(0, len(mi)):
            j_ax = a_ax01[mi[j][0], mi[j][1]]
            for i in range(0, n_levels):
                row_start = 3 * n_T_htf + i * n_T_amb
                row_end = row_start + n_T_amb
                if( j == 0 ):
                    j_ax.plot([k[2] for k in local_udpc_data[row_start:row_end]],
                        [k[mi[j][2]] for k in local_udpc_data[row_start:row_end]],lcolor[i]+lstyle[i_udpc],
                            label = "T_HTF, Case " + str(i_udpc+1) + " = " + str(local_udpc_data[row_start][0]))

                else:
                    j_ax.plot([k[2] for k in local_udpc_data[row_start:row_end]],
                            [k[mi[j][2]] for k in local_udpc_data[row_start:row_end]],lcolor[i]+lstyle[i_udpc])
            
            if(i_udpc == 0):
                j_ax.set_xlabel("Ambient Temperature [C]")
                j_ax.set_ylabel(mi[j][3])
                j_ax.grid(which='both', color='gray', alpha=1)

    fig01.legend(ncol=n_udpcs, loc="upper center", columnspacing=0.6, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(pad=0.0, h_pad=1, w_pad=w_pad, rect=(0.012, 0.02, 0.98, 0.86))
    plt.savefig('results/' + plot_pre_str + "udpc_comp_T_amb.png")
    plt.close()


    fig2, a_ax2 = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(7, f_h))

    for i_udpc, local_udpc_data in enumerate(list_of_udpc_data):

        # m_dot parametric values, 3 T_amb levels, design T_htf_hot
        for j in range(0, len(mi)):
            j_ax = a_ax2[mi[j][0], mi[j][1]]
            for i in range(0, n_levels):
                row_start = 3 * n_T_htf + 3 * n_T_amb + i * n_m_dot_htf
                row_end = row_start + n_m_dot_htf
                if( j == 0 ):
                    j_ax.plot([k[1] for k in local_udpc_data[row_start:row_end]],
                        [k[mi[j][2]] for k in local_udpc_data[row_start:row_end]],lcolor[i]+lstyle[i_udpc],
                        label = "T_amb, Case " + str(i_udpc+1) + " = " + str(local_udpc_data[row_start][2]))

                else:
                    j_ax.plot([k[1] for k in local_udpc_data[row_start:row_end]],
                        [k[mi[j][2]] for k in local_udpc_data[row_start:row_end]],lcolor[i]+lstyle[i_udpc],)

            if(i_udpc == 0):
                j_ax.set_xlabel("Normalized HTF Mass Flow")
                j_ax.set_ylabel(mi[j][3])
                j_ax.grid(which='both', color='gray', alpha=1)

    fig2.legend(ncol = n_udpcs, loc = "upper center", columnspacing = 0.6, bbox_to_anchor = (0.5,1.0))
    plt.tight_layout(pad=0.0, h_pad=1, w_pad=w_pad, rect=(0.012, 0.02, 0.98, 0.86))
    plt.savefig('results/' + plot_pre_str + "udpc_comp_m_dot_htf.png")
    plt.close()

       
def make_udpc_plots_from_json_dict(json_file_name):

    udpc_dict = json.load(open(json_file_name))

    print("HTF cold design = " + str(udpc_dict["T_htf_cold_des"]) + " C")

    T_hot_str = "HTF Hot Temperature (Design page) = " + str(udpc_dict["T_htf_hot_des"]) + " C"
    T_cold_str = "HTF Cold Temperature (Design page) = " + str(udpc_dict["T_htf_cold_des"]) + " C"
    eta_str = "Cycle Thermal Efficiency (Design page) = " + str(udpc_dict["eta_thermal_calc"]) + " -"
    T_amb_str = "Ambient Temperature (Power Cycle page) = " + str(udpc_dict["T_amb_des"]) + " C"
    W_dot_cool_str = "Cooling Parasitic (Power Cycle page) = " + str(udpc_dict["fan_power_frac"]) + " -"

    od_T_t_in_mode = udpc_dict["od_T_t_in_mode"]

    n_T_htf = int(udpc_dict["udpc_n_T_htf"])
    n_T_amb = int(udpc_dict["udpc_n_T_amb"])
    n_m_dot_htf = int(udpc_dict["udpc_n_m_dot_htf"])

    udpc_data = udpc_dict["udpc_table"]

    s_cycle_des = T_hot_str + "\n" + T_cold_str + "\n" + eta_str + "\n" + T_amb_str + "\n" + W_dot_cool_str + "\n"

    plot_udpc_results(udpc_data, n_T_htf, n_T_amb, n_m_dot_htf, "updc_data_read", s_cycle_des, od_T_t_in_mode)       
    

def make_compare_udpc_plots_from_jsons(json_file_name1, json_file_name2):

    udpc_dict1 = json.load(open(json_file_name1))
    udpc_dict2 = json.load(open(json_file_name2))

    n_T_htf = int(udpc_dict1["udpc_n_T_htf"])
    n_T_amb = int(udpc_dict1["udpc_n_T_amb"])
    n_m_dot_htf = int(udpc_dict1["udpc_n_m_dot_htf"])

    udpc_data1 = udpc_dict1["udpc_table"]
    udpc_data2 = udpc_dict2["udpc_table"]

    plot_compare_udpc_results([udpc_data1, udpc_data2], n_T_htf, n_T_amb, n_m_dot_htf, "updc_data_comp")
     
        
        
        
        
        
        
