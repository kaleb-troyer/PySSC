
import matplotlib as mpl
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'  # 'azel', 'trackball', 'sphere', or 'arcball'

from labellines import labelLines
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d as interp
from matplotlib.tri import Triangulation as tri
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import addcopyfighandler
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import types
import time
import os
import gc

# for benchmarking the program speed
class timer():
    def __init__(self, quiet=False):
        self.start = 0
        self.bench = 0
        self.quiet = quiet
        self.timed = False
        self.lap = 0
    def tic(self):
        if self.timed == 0:
            self.timed = True
            self.start = time.time()
            if not self.quiet: print("\nStarting timer...")
        else: 
            self.lap = time.time()
            if not self.quiet: print(f"{self.lap - self.start:.4f}")
            return self.lap - self.start
    def toc(self):
        if self.timed:
            self.timed = False
            self.bench = time.time()
            if not self.quiet: print(f"time elapsed: {self.bench-self.start:.4f}")
        else: pass
        return self.bench-self.start

class Struct():
    '''
    Custom dataclass for design and solution parameters. The Plotter
    class expects parameters of this typing. The dataclass is not 
    extensible and only the 'value' member is mutable. 

    ### Attributes

    **key**: `str=''`  
    SSC string name for parameter.  

    **repr**: `str=''`  
    Description of parameter.  

    **units**: `str=''`  
    LaTeX string representation of the parameter units.  

    **dtype**: `str=''`  
    String representation of the parameter data type.  

    **default**: `float=0.0`  
    Default value of the parameter.  

    **value**: `float=0.0`  
    *(optional)* Mutable value of the parameter.  
    '''

    __slots__ = ('key', 'repr', 'units', 'dtype', 'acro', 'default', 'value')  # Restricts attributes and reduces memory usage

    def __init__(self, key: str=None, repr: str=None, units: str=None, dtype: str=None, acro: str=None, default: float=None, value: float=None):

        super().__setattr__('key', key)
        super().__setattr__('repr', repr)
        super().__setattr__('units', units)
        super().__setattr__('dtype', dtype)
        super().__setattr__('default', default)
        super().__setattr__('value', value)
        super().__setattr__('acro', acro)

    def __setattr__(self, name, value):
        _frozen = ('key', 'repr', 'units', 'dtype', 'default')
        _mutable = ('value')
        if name in _mutable: 
            super().__setattr__(name, value)
        elif name in _frozen: 
            raise AttributeError(f"Cannot modify a frozen attribute ('{name}').")
        else: raise AttributeError(f"Attribute unknown ('{name}'). Struct is immutable.")

    def __repr__(self):
        return f"Struct(key={self.key}, repr={self.repr}, units={self.units}, dtype={self.dtype})"

class Parameters(): 
    '''
    Serves as a data container for all SSC design and solution 
    parameters. Parameters are of the custom Struct type, a dataclass
    with variables for the parameter key, data type, string representation, 
    and units. A custom dataclass is used so that parameters are accessible
    via intellisense. 

    Example
    ``` 
    par = Parameters()
    par.cycle_design.key
    >>> "design_method"

    par.cycle_design.dtype
    >>> "int64"

    par.cycle_design.repr
    >>> "selects the cycle design method"

    par.cycle_design.units
    >>> "[-]"
    ```

    ### Methods
    ---
    ```
    get() -> list
        Provides a list of all parameters as Structs.
    ```
    '''

    def __init__(self):
        self._despar()
        self._dessol()
    def get(self): 
        return [value for _, value in vars(self).items() if isinstance(value, dict)]
    
    def _despar(self): 
        self.quiet          = Struct(key = "quiet",                 dtype = "bool",     repr = "no log notices",                                 units = "[-]"          )
        self.opt_logging    = Struct(key = "opt_logging",           dtype = "bool",     repr = "log each optimization result",                   units = "[-]"          )
        self.opt_penalty    = Struct(key = "opt_penalty",           dtype = "bool",     repr = "apply penalties to optimization",                units = "[-]"          )
        self.try_s_cycle    = Struct(key = "try_simple_cycle",      dtype = "bool",     repr = "check simple cycles",                            units = "[-]"          )
        self.try_r_cycle    = Struct(key = "is_recomp_ok",          dtype = "float64",  repr = "fix or optimize recompression fraction",         units = "[-]"          )
        self.fix_P_high     = Struct(key = "is_P_high_fixed",       dtype = "bool",     repr = "fix or optimize high pressure",                  units = "[-]"          )
        self.fix_P_ratio    = Struct(key = "is_PR_fixed",           dtype = "float64",  repr = "fix or optimize pressure ratio",                 units = "[-]"          )
        self.fix_P_intermed = Struct(key = "is_IP_fixed",           dtype = "float64",  repr = "fix or optimize intermediate pressure",          units = "[-]"          )
        self.cycle_config   = Struct(key = "cycle_config",          dtype = "int64",    repr = "selects recompression or precompression cycle",  units = "[-]"          )
        self.cycle_design   = Struct(key = "design_method",         dtype = "int64",    repr = "selects the cycle design method",                units = "[-]"          )
        self.opt_objective  = Struct(key = "des_objective",         dtype = "int64",    repr = "optimization objective",                         units = "[-]"          )
        self.opt_tolerance  = Struct(key = "rel_tol",               dtype = "float64",  repr = "optimization tolerance exponent",                units = "[-]"          )
        self.htf_code       = Struct(key = "htf",                   dtype = "int64",    repr = "Heat Transfer Fluid",                            units = "[-]"          )
        self.PHX_hot_in     = Struct(key = "T_htf_hot_des",         dtype = "float64",  repr = "PHX Hot Inlet Temperature",                      units = "[C]"          )
        self.ambient_temp   = Struct(key = "T_amb_des",             dtype = "float64",  repr = "Ambient Design Temperature",                     units = "[C]"          )
        self.dT_mc_approach = Struct(key = "dT_mc_approach",        dtype = "float64",  repr = "Compressor Approach Temperature",                units = "[C]"          )
        self.site_elevation = Struct(key = "site_elevation",        dtype = "float64",  repr = "Site Elevation",                                 units = "[m]"          )
        self.W_dot_net      = Struct(key = "W_dot_net_des",         dtype = "float64",  repr = "Design Cycle Power Output",                      units = "[MWe]"        )
        self.TES_capacity   = Struct(key = "TES_capacity",          dtype = "float64",  repr = "Thermal Energy Storage",                         units = "[hours]"      )
        self.heliostat_cost = Struct(key = "heliostat_cost",        dtype = "float64",  repr = "Heliostat Reflective Area Cost",                 units = r"[$\$/m^{2}$]")
        self.rec_eta_mod    = Struct(key = "receiver_eta_mod",      dtype = "float64",  repr = "Receiver Efficiency Modifier",                   units = "[-]"          )
        self.eta_thermal    = Struct(key = "eta_thermal_des",       dtype = "float64",  repr = "target cycle efficiency, if applicable",         units = "[-]"          )
        self.UA_recup_tot   = Struct(key = "UA_recup_tot_des",      dtype = "float64",  repr = "Conductance Allocated to Recuperators",          units = "[kW/K]"       )
        self.eta_isen_mc    = Struct(key = "eta_isen_mc",           dtype = "float64",  repr = "Main Compressor Isentropic Efficiency",          units = "[-]"          )
        self.eta_isen_rc    = Struct(key = "eta_isen_rc",           dtype = "float64",  repr = "Recompressor Isentropic Efficiency",             units = "[-]"          )
        self.eta_isen_pc    = Struct(key = "eta_isen_pc",           dtype = "float64",  repr = "Precompressor Isentropic Efficiency",            units = "[-]"          )
        self.gross_to_net   = Struct(key = "gross_to_net",          dtype = "float64",  repr = "Electric Generator Efficiency",                  units = "[-]"          )
        self.eta_isen_t     = Struct(key = "eta_isen_t",            dtype = "float64",  repr = "Turbine Isentropic Efficiency",                  units = "[-]"          )
        self.P_high_limit   = Struct(key = "P_high_limit",          dtype = "float64",  repr = "Cycle Pressure Limit",                           units = "[-]"          )
        self.LTR_code       = Struct(key = "LTR_design_code",       dtype = "int64",    repr = "LTR design method",                              units = "[-]"          )
        self.LTR_UA         = Struct(key = "LTR_UA_des_in",         dtype = "float64",  repr = "UA Allocated to LTR",                            units = "[kW/K]"       )
        self.LTR_dT_min     = Struct(key = "LTR_min_dT_des_in",     dtype = "float64",  repr = "LTR Minimum Approach Temperature",               units = "[C]"          )
        self.LTR_eff        = Struct(key = "LTR_eff_des_in",        dtype = "float64",  repr = "LTR Effectiveness",                              units = "[-]"          )
        self.LTR_eff_max    = Struct(key = "LT_recup_eff_max",      dtype = "float64",  repr = "LTR Maximum Effectiveness",                      units = "[-]"          )
        self.LTR_LP_loss    = Struct(key = "LTR_LP_deltaP_des_in",  dtype = "float64",  repr = "LTR LP-Side Pressure Loss Fraction",             units = "[-]"          )
        self.LTR_HP_loss    = Struct(key = "LTR_HP_deltaP_des_in",  dtype = "float64",  repr = "LTR HP-Side Pressure Loss Fraction",             units = "[-]"          )
        self.HTR_code       = Struct(key = "HTR_design_code",       dtype = "int64",    repr = "HTR Design Method",                              units = "[-]"          )
        self.HTR_UA         = Struct(key = "HTR_UA_des_in",         dtype = "float64",  repr = "UA Allocated to HTR",                            units = "[kW/K]"       )
        self.HTR_dT_min     = Struct(key = "HTR_min_dT_des_in",     dtype = "float64",  repr = "HTR Minimum Approach Temperature",               units = "[C]"          )
        self.HTR_eff        = Struct(key = "HTR_eff_des_in",        dtype = "float64",  repr = "HTR Effectiveness",                              units = "[-]"          )
        self.HTR_eff_max    = Struct(key = "HT_recup_eff_max",      dtype = "float64",  repr = "HTR Maximum Effectiveness",                      units = "[-]"          )
        self.HTR_LP_loss    = Struct(key = "HTR_LP_deltaP_des_in",  dtype = "float64",  repr = "HTR LP-Side Pressure Loss Fraction",             units = "[-]"          )
        self.HTR_HP_loss    = Struct(key = "HTR_HP_deltaP_des_in",  dtype = "float64",  repr = "HTR HP-Side Pressure Loss Fraction",             units = "[-]"          )
        self.PHX_HP_loss    = Struct(key = "PHX_co2_deltaP_des_in", dtype = "float64",  repr = "PHX Pressure Loss Fraction",                     units = "[-]"          )
        self.PHX_dT_hot     = Struct(key = "dT_PHX_hot_approach",   dtype = "float64",  repr = "dT (PHX Hot Approach)",                          units = "[C]"          )
        self.PHX_dT_cold    = Struct(key = "dT_PHX_cold_approach",  dtype = "float64",  repr = "dT (PHX Cold Approach)",                         units = "[C]"          )
        self.PHX_subs       = Struct(key = "PHX_n_sub_hx",          dtype = "float64",  repr = "PHX subdivisions used in design calculation",    units = "[-]"          )
        self.PHX_cost_basis = Struct(key = "PHX_cost_model",        dtype = "float64",  repr = "PHX Cost Basis",                                 units = "[%]"          )
        self.AC_LP_loss     = Struct(key = "deltaP_cooler_frac",    dtype = "float64",  repr = "Air Cooler Pressure Loss Fraction",              units = "[-]"          )
        self.AC_parasitics  = Struct(key = "fan_power_frac",        dtype = "float64",  repr = "Fraction of Net Cycle Power Consumed by Cooler", units = "[-]"          )
        self.HX_p_loss_frac = Struct(key = "deltaP_counterHX_frac", dtype = "float64",  repr = "Counterflow HX Pressure Loss Fraction",          units = "[-]"          )
    def _dessol(self):
        self.T_htf_cold_des                 = Struct(key = "T_htf_cold_des",                   dtype = "float64", repr = "HTF design cold temperature (PHX outlet)",                         units = r"[C]",          )
        self.m_dot_htf_des                  = Struct(key = "m_dot_htf_des",                    dtype = "float64", repr = "HTF mass flow rate",                                               units = r"[kg/s]",       )
        self.V_dot_htf_des                  = Struct(key = "V_dot_htf_des",                    dtype = "float64", repr = "HTF volumetric flow rate",                                         units = r"[m3/s]",       )
        self.eta_thermal_calc               = Struct(key = "eta_thermal_calc",                 dtype = "float64", repr = "Power Cycle Thermal Efficiency",                                   units = r"[-]",          )
        self.m_dot_co2_full                 = Struct(key = "m_dot_co2_full",                   dtype = "float64", repr = "CO2 Mass Flow Rate",                                               units = r"[kg/s]",       )
        self.recomp_frac                    = Struct(key = "recomp_frac",                      dtype = "float64", repr = "Recompression Fraction",                                           units = r"[-]",          )
        self.cycle_cost                     = Struct(key = "cycle_cost",                       dtype = "float64", repr = "Cycle cost bare erected",                                          units = r"[M$]",         )
        self.cycle_spec_cost                = Struct(key = "cycle_spec_cost",                  dtype = "float64", repr = "Cycle specific cost bare erected",                                 units = r"[$/kWe]",      )
        self.cycle_spec_cost_thermal        = Struct(key = "cycle_spec_cost_thermal",          dtype = "float64", repr = "Cycle specific (thermal) cost bare erected",                       units = r"[$/kWt]",      )
        self.W_dot_net_less_cooling         = Struct(key = "W_dot_net_less_cooling",           dtype = "float64", repr = "System power output subtracting cooling parastics",                units = r"[MWe]"         )
        self.rec_eta                        = Struct(key = "receiver_efficiency",              dtype = "float64", repr = "Receiver Thermal Efficiency",                                      units = r"[%]",          )
        self.total_cost                     = Struct(key = "total_cost",                       dtype = "float64", repr = "Total cost of CSP and power cycle",                                units = r"[M$]",         )
        self.total_spec_cost                = Struct(key = "total_spec_cost",                  dtype = "float64", repr = "Total specific cost bare erected",                                 units = r"[M$/kWe]",     )
        self.total_spec_cost_thermal        = Struct(key = "total_spec_cost_thermal",          dtype = "float64", repr = "Total specific (thermal) cost bare erected",                       units = r"[M$/kWt]",     )
        self.solar_tower_cost               = Struct(key = "solar_tower_cost",                 dtype = "float64", repr = "CSP Gen3 Solar Tower capital cost",                                units = r"[M$]",       acro = r"Solar Tower"       )
        self.solar_field_cost               = Struct(key = "solar_field_cost",                 dtype = "float64", repr = "CSP Gen3 Solar Field capital cost",                                units = r"[M$]",       acro = r"Solar Field"       )
        self.receiver_cost                  = Struct(key = "falling_particle_receiver",        dtype = "float64", repr = "CSP Gen3 Falling Particle Receiver capital cost",                  units = r"[M$]",       acro = r"Receiver"          )
        self.particles_cost                 = Struct(key = "particles_cost",                   dtype = "float64", repr = "bulk cost of particles",                                           units = r"[M$]",       acro = r"Particles"         )
        self.particle_losses_cost           = Struct(key = "particle_losses_cost",             dtype = "float64", repr = "incurred cost due to particle loss / attrition",                   units = r"[M$]",       acro = r"Attrition"         )
        self.particle_storage_cost          = Struct(key = "particle_storage_cost",            dtype = "float64", repr = "particle storage bins and insulation capital cost",                units = r"[M$]",       acro = r"TES"               )
        self.particle_lifts_cost            = Struct(key = "particle_lifts_cost",              dtype = "float64", repr = "particle transportation capital cost",                             units = r"[M$]",       acro = r"Lifts"             )
        self.land_cost                      = Struct(key = "land_cost",                        dtype = "float64", repr = "bulk cost of land required for power plant",                       units = r"[M$]",       acro = r"Land"              )
        self.HTR_capital_cost               = Struct(key = "HTR_capital_cost",                 dtype = "float64", repr = "high temperature recuperator capital cost",                        units = r"[M$]",       acro = r"HTR"               )
        self.LTR_capital_cost               = Struct(key = "LTR_capital_cost",                 dtype = "float64", repr = "low temperature recuperator capital cost",                         units = r"[M$]",       acro = r"LTR"               )
        self.PHX_capital_cost               = Struct(key = "PHX_capital_cost",                 dtype = "float64", repr = "primary heat exchanger capital cost",                              units = r"[M$]",       acro = r"PHX"               )
        self.air_cooler_capital_cost        = Struct(key = "air_cooler_capital_cost",          dtype = "float64", repr = "air cooler capital cost",                                          units = r"[M$]",       acro = r"Air Cooler"        )
        self.compressor_capital_cost        = Struct(key = "compressor_capital_cost",          dtype = "float64", repr = "primary compressor capital cost",                                  units = r"[M$]",       acro = r"Compressor"        )
        self.recompressor_capital_cost      = Struct(key = "recompressor_capital_cost",        dtype = "float64", repr = "recompressor capital cost",                                        units = r"[M$]",       acro = r"Recompressor"      )
        self.turbine_capital_cost           = Struct(key = "turbine_capital_cost",             dtype = "float64", repr = "turbine capital cost",                                             units = r"[M$]",       acro = r"Turbine"           )
        self.piping_capital_cost            = Struct(key = "piping_capital_cost",              dtype = "float64", repr = "piping, inventory control, etc.",                                  units = r"[M$]",       acro = r"Piping, etc."      )
        self.piping_cost_factor             = Struct(key = "piping_cost_factor",               dtype = "float64", repr = "% of cycle capital constituting pipe costs",                       units = r"[M$]",       acro = r"Piping Fraction"   )
        self.balance_of_plant_cost          = Struct(key = "balance_of_plant_cost",            dtype = "float64", repr = "transformers, inverters, controls, etc.",                          units = r"[M$]",       acro = r"BOP"               )
        self.cycle_capital_cost             = Struct(key = "cycle_capital_cost",               dtype = "float64", repr = "Power block capital costs",                                        units = r"[M$]",       acro = r"Power Block"       )
        self.plant_capital_cost             = Struct(key = "plant_capital_cost",               dtype = "float64", repr = "CSP equipment capital costs",                                      units = r"[M$]",       acro = r"CSP Components"    )
        self.total_capital_cost             = Struct(key = "total_capital_cost",               dtype = "float64", repr = "total expected capital cost of plant",                             units = r"[M$]",       acro = r"Total"             )
        self.annual_maintenance_cost        = Struct(key = "annual_maintenance_cost",          dtype = "float64", repr = "expected O&M annual costs",                                        units = r"[M$/year]",  acro = r"O&M"               )
        self.total_adjusted_cost            = Struct(key = "total_adjusted_cost",              dtype = "float64", repr = "Total Adjusted Cost",                                              units = r"[M$]",       acro = r"Total Adj."        )
        self.levelized_cost_of_energy       = Struct(key = "levelized_cost_of_energy",         dtype = "float64", repr = "Levelized Cost of Energy",                                         units = r"[$/MWe-h]",  acro = r"LCOE"              )
        self.T_comp_in                      = Struct(key = "T_comp_in",                        dtype = "float64", repr = "Compressor Inlet Temperature",                                     units = r"[C]",          )
        self.P_comp_in                      = Struct(key = "P_comp_in",                        dtype = "float64", repr = "Compressor Inlet Pressure",                                        units = r"[MPa]",        )
        self.P_comp_out                     = Struct(key = "P_comp_out",                       dtype = "float64", repr = "Compressor Outlet Pressure",                                       units = r"[MPa]",        )
        self.mc_T_out                       = Struct(key = "mc_T_out",                         dtype = "float64", repr = "Compressor Outlet Temperature",                                    units = r"[C]",          )
        self.mc_W_dot                       = Struct(key = "mc_W_dot",                         dtype = "float64", repr = "Compressor Power",                                                 units = r"[MWe]",        )
        self.mc_m_dot_des                   = Struct(key = "mc_m_dot_des",                     dtype = "float64", repr = "Compressor mass flow rate",                                        units = r"[kg/s]",       )
        self.mc_rho_in                      = Struct(key = "mc_rho_in",                        dtype = "float64", repr = "Compressor Inlet Density",                                         units = r"[kg/m3]",      )
        self.mc_ideal_spec_work             = Struct(key = "mc_ideal_spec_work",               dtype = "float64", repr = "Compressor Ideal Spec Work",                                       units = r"[kJ/kg]",      )
        self.mc_phi_des                     = Struct(key = "mc_phi_des",                       dtype = "float64", repr = "Compressor design flow coefficient",                               units = r"[-]",          )
        self.mc_psi_d                       = Struct(key = "mc_psi_des",                       dtype = "float64", repr =  "Compressor design ideal head coefficient",                        units = r"[-]",          )
        self.mc_tip_ratio_des               = Struct(key = "mc_tip_ratio_des",                 dtype = "object",  repr = "Compressor design stage tip speed ratio",                          units = r"[-]",          )
        self.mc_n_stages                    = Struct(key = "mc_n_stages",                      dtype = "float64", repr = "Compressor stages",                                                units = r"[-]",          )
        self.mc_N_des                       = Struct(key = "mc_N_des",                         dtype = "float64", repr = "Compressor design shaft speed",                                    units = r"[rpm]",        )
        self.mc_D                           = Struct(key = "mc_D",                             dtype = "object",  repr = "Compressor stage diameters",                                       units = r"[m]",          )
        self.mc_phi_surge                   = Struct(key = "mc_phi_surge",                     dtype = "float64", repr = "Compressor flow coefficient where surge occurs",                   units = r"[-]",          )
        self.mc_psi_max_at_N_des            = Struct(key = "mc_psi_max_at_N_des",              dtype = "float64", repr = "Compressor max ideal head coefficient at design shaft speed",      units = r"[-]",          )
        self.mc_eta_stages_des              = Struct(key = "mc_eta_stages_des",                dtype = "object",  repr = "Compressor design stage isentropic efficiencies",                  units = r"[-]",          )
        self.mc_cost_equipment              = Struct(key = "mc_cost_equipment",                dtype = "float64", repr = "Compressor cost equipment",                                        units = r"[M$]",         )
        self.mc_cost_bare_erected           = Struct(key = "mc_cost_bare_erected",             dtype = "float64", repr = "Compressor cost equipment plus install",                           units = r"[M$]",         )
        self.rc_T_in_des                    = Struct(key = "rc_T_in_des",                      dtype = "float64", repr = "Recompressor inlet temperature",                                   units = r"[C]",          )
        self.rc_P_in_des                    = Struct(key = "rc_P_in_des",                      dtype = "float64", repr = "Recompressor Inlet Pressure",                                      units = r"[MPa]",        )
        self.rc_T_out_des                   = Struct(key = "rc_T_out_des",                     dtype = "float64", repr = "Recompressor Inlet Pemperature",                                   units = r"[C]",          )
        self.rc_P_out_des                   = Struct(key = "rc_P_out_des",                     dtype = "float64", repr = "Recompressor Inlet Pressure",                                      units = r"[MPa]",        )
        self.rc_W_dot                       = Struct(key = "rc_W_dot",                         dtype = "float64", repr = "Recompressor power",                                               units = r"[MWe]",        )
        self.rc_m_dot_des                   = Struct(key = "rc_m_dot_des",                     dtype = "float64", repr = "Recompressor mass flow rate",                                      units = r"[kg/s]",       )
        self.rc_phi_des                     = Struct(key = "rc_phi_des",                       dtype = "float64", repr = "Recompressor design flow coefficient",                             units = r"[-]",          )
        self.rc_psi_des                     = Struct(key = "rc_psi_des",                       dtype = "float64", repr = "Recompressor design ideal head coefficient",                       units = r"[-]",          )
        self.rc_tip_ratio_des               = Struct(key = "rc_tip_ratio_des",                 dtype = "object",  repr = "Recompressor design stage tip speed ratio",                        units = r"[-]",          )
        self.rc_n_stages                    = Struct(key = "rc_n_stages",                      dtype = "float64", repr = "Recompressor stages",                                              units = r"[-]",          )
        self.rc_N_des                       = Struct(key = "rc_N_des",                         dtype = "float64", repr = "Recompressor design shaft speed",                                  units = r"[rpm]",        )
        self.rc_D                           = Struct(key = "rc_D",                             dtype = "object",  repr = "Recompressor stage diameters",                                     units = r"[m]",          )
        self.rc_phi_surge                   = Struct(key = "rc_phi_surge",                     dtype = "float64", repr = "Recompressor flow coefficient where surge occurs",                 units = r"[-]",          )
        self.rc_psi_max_at_N_des            = Struct(key = "rc_psi_max_at_N_des",              dtype = "float64", repr = "Recompressor max ideal head coefficient at design shaft speed",    units = r"[-]",          )
        self.rc_eta_stages_des              = Struct(key = "rc_eta_stages_des",                dtype = "object",  repr = "Recompressor design stage isenstropic efficiencies",               units = r"[-]",          )
        self.rc_cost_equipment              = Struct(key = "rc_cost_equipment",                dtype = "float64", repr = "Recompressor cost equipment",                                      units = r"[M$]",         )
        self.rc_cost_bare_erected           = Struct(key = "rc_cost_bare_erected",             dtype = "float64", repr = "Recompressor cost equipment plus install",                         units = r"[M$]",         )
        self.pc_T_in_des                    = Struct(key = "pc_T_in_des",                      dtype = "float64", repr = "Precompressor inlet temperature",                                  units = r"[C]",          )
        self.pc_P_in_des                    = Struct(key = "pc_P_in_des",                      dtype = "float64", repr = "Precompressor inlet pressure",                                     units = r"[MPa]",        )
        self.pc_W_dot                       = Struct(key = "pc_W_dot",                         dtype = "float64", repr = "Precompressor power",                                              units = r"[MWe]",        )
        self.pc_m_dot_des                   = Struct(key = "pc_m_dot_des",                     dtype = "float64", repr = "Precompressor mass flow rate",                                     units = r"[kg/s]",       )
        self.pc_rho_in_des                  = Struct(key = "pc_rho_in_des",                    dtype = "float64", repr = "Precompressor inlet density",                                      units = r"[kg/$m^{3}$]", )
        self.pc_ideal_spec_work_des         = Struct(key = "pc_ideal_spec_work_des",           dtype = "float64", repr = "Precompressor ideal spec work",                                    units = r"[kJ/kg]",      )
        self.pc_phi_des                     = Struct(key = "pc_phi_des",                       dtype = "float64", repr = "Precompressor design flow coefficient",                            units = r"[-]",          )
        self.pc_tip_ratio_des               = Struct(key = "pc_tip_ratio_des",                 dtype = "object",  repr = "Precompressor design stage tip speed ratio",                       units = r"[-]",          )
        self.pc_n_stages                    = Struct(key = "pc_n_stages",                      dtype = "float64", repr = "Precompressor stages",                                             units = r"[-]",          )
        self.pc_N_des                       = Struct(key = "pc_N_des",                         dtype = "float64", repr = "Precompressor design shaft speed",                                 units = r"[rpm]",        )
        self.pc_D                           = Struct(key = "pc_D",                             dtype = "object",  repr = "Precompressor stage diameters",                                    units = r"[m]",          )
        self.pc_phi_surge                   = Struct(key = "pc_phi_surge",                     dtype = "float64", repr = "Precompressor flow coefficient where surge occurs",                units = r"[-]",          )
        self.pc_eta_stages_des              = Struct(key = "pc_eta_stages_des",                dtype = "object",  repr = "Precompressor design stage isenstropic efficiencies",              units = r"[-]",          )
        self.pc_cost_equipment              = Struct(key = "pc_cost_equipment",                dtype = "float64", repr = "Precompressor cost equipment",                                     units = r"[M$]",         )
        self.pc_cost_bare_erected           = Struct(key = "pc_cost_bare_erected",             dtype = "float64", repr = "Precompressor cost equipment plus install",                        units = r"[M$]",         )
        self.c_tot_cost_equip               = Struct(key = "c_tot_cost_equip",                 dtype = "float64", repr = "Compressor total cost",                                            units = r"[M$]",         )
        self.c_tot_W_dot                    = Struct(key = "c_tot_W_dot",                      dtype = "float64", repr = "Compressor total summed power",                                    units = r"[MWe]",        )
        self.t_W_dot                        = Struct(key = "t_W_dot",                          dtype = "float64", repr = "Turbine power",                                                    units = r"[MWe]",        )
        self.t_m_dot_des                    = Struct(key = "t_m_dot_des",                      dtype = "float64", repr = "Turbine mass flow rate",                                           units = r"[kg/s]",       )
        self.T_turb_in                      = Struct(key = "T_turb_in",                        dtype = "float64", repr = "Turbine inlet temperature",                                        units = r"[C]",          )
        self.t_P_in_des                     = Struct(key = "t_P_in_des",                       dtype = "float64", repr = "Turbine design inlet pressure",                                    units = r"[MPa]",        )
        self.t_T_out_des                    = Struct(key = "t_T_out_des",                      dtype = "float64", repr = "Turbine outlet temperature",                                       units = r"[C]",          )
        self.t_P_out_des                    = Struct(key = "t_P_out_des",                      dtype = "float64", repr = "Turbine design outlet pressure",                                   units = r"[MPa]",        )
        self.t_specific_work                = Struct(key = "t_delta_h_isen_des",               dtype = "float64", repr = "Turbine Isentropic Specific Work",                                 units = r"[kJ/kg]",      )
        self.t_rho_in_des                   = Struct(key = "t_rho_in_des",                     dtype = "float64", repr = "Turbine inlet density",                                            units = r"[kg/$m^{3}$]", )
        self.t_nu_des                       = Struct(key = "t_nu_des",                         dtype = "float64", repr = "Turbine design velocity ratio",                                    units = r"[-]",          )
        self.t_tip_ratio_des                = Struct(key = "t_tip_ratio_des",                  dtype = "float64", repr = "Turbine design tip speed ratio",                                   units = r"[-]",          )
        self.t_N_des                        = Struct(key = "t_N_des",                          dtype = "float64", repr = "Turbine design shaft speed",                                       units = r"[rpm]",        )
        self.t_D                            = Struct(key = "t_D",                              dtype = "float64", repr = "Turbine diameter",                                                 units = r"[m]",          )
        self.t_cost_equipment               = Struct(key = "t_cost_equipment",                 dtype = "float64", repr = "Tubine cost - equipment",                                          units = r"[M$]",         )
        self.t_cost_bare_erected            = Struct(key = "t_cost_bare_erected",              dtype = "float64", repr = "Tubine cost - equipment plus install",                             units = r"[M$]",         )
        self.recup_total_UA_assigned        = Struct(key = "recup_total_UA_assigned",          dtype = "float64", repr = "Total recuperator UA assigned to design routine",                  units = r"[MW/K]",       )
        self.recup_total_cost_equipment     = Struct(key = "recup_total_cost_equipment",       dtype = "float64", repr = "Total recuperator cost equipment",                                 units = r"[M$]",         )
        self.recup_total_cost_bare_erected  = Struct(key = "recup_total_cost_bare_erected",    dtype = "float64", repr = "Total recuperator cost bare erected",                              units = r"[M$]",         )
        self.recup_LTR_UA_frac              = Struct(key = "recup_LTR_UA_frac",                dtype = "float64", repr = "Fraction of total conductance to LTR",                             units = r"[-]",          )
        self.LTR_HP_T_out_des               = Struct(key = "LTR_HP_T_out_des",                 dtype = "float64", repr = "Low temp recuperator HP outlet temperature",                       units = r"[C]",          )
        self.LTR_UA_assigned                = Struct(key = "LTR_UA_assigned",                  dtype = "float64", repr = "Low temp recuperator UA assigned from total",                      units = r"[MW/K]",       ) 
        self.eff_LTR                        = Struct(key = "eff_LTR",                          dtype = "float64", repr = "Low temp recuperator effectiveness",                               units = r"[-]",          )
        self.NTU_LTR                        = Struct(key = "NTU_LTR",                          dtype = "float64", repr = "Low temp recuperator NTU",                                         units = r"[-]",          )
        self.q_dot_LTR                      = Struct(key = "q_dot_LTR",                        dtype = "float64", repr = "Low temp recuperator heat transfer",                               units = r"[MWt]",        )
        self.LTR_LP_deltaP_des              = Struct(key = "LTR_LP_deltaP_des",                dtype = "float64", repr = "Low temp recuperator low pressure design pressure drop",           units = r"[-]",          )
        self.LTR_HP_deltaP_des              = Struct(key = "LTR_HP_deltaP_des",                dtype = "float64", repr = "Low temp recuperator high pressure design pressure drop",          units = r"[-]",          )
        self.LTR_min_dT                     = Struct(key = "LTR_min_dT",                       dtype = "float64", repr = "Low temp recuperator min temperature difference",                  units = r"[C]",          )
        self.LTR_cost_equipment             = Struct(key = "LTR_cost_equipment",               dtype = "float64", repr = "Low temp recuperator cost equipment",                              units = r"[M$]",         )
        self.LTR_cost_bare_erected          = Struct(key = "LTR_cost_bare_erected",            dtype = "float64", repr = "Low temp recuperator cost equipment and install",                  units = r"[M$]",         )
        self.HTR_LP_T_out_des               = Struct(key = "HTR_LP_T_out_des",                 dtype = "float64", repr = "High temp recuperator LP outlet temperature",                      units = r"[C]",          )
        self.HTR_HP_T_in_des                = Struct(key = "HTR_HP_T_in_des",                  dtype = "float64", repr = "High temp recuperator HP inlet temperature",                       units = r"[C]",          )
        self.HTR_UA_assigned                = Struct(key = "HTR_UA_assigned",                  dtype = "float64", repr = "High temp recuperator UA assigned from total",                     units = r"[MW/K]",       )
        self.eff_HTR                        = Struct(key = "eff_HTR",                          dtype = "float64", repr = "High temp recuperator effectiveness",                              units = r"[-]",          )
        self.NTU_HTR                        = Struct(key = "NTU_HTR",                          dtype = "float64", repr = "High temp recuperator NTRU",                                       units = r"[-]",          )
        self.q_dot_HTR                      = Struct(key = "q_dot_HTR",                        dtype = "float64", repr = "High temp recuperator heat transfer",                              units = r"[MWt]",        )
        self.HTR_LP_deltaP_des              = Struct(key = "HTR_LP_deltaP_des",                dtype = "float64", repr = "High temp recuperator low pressure design pressure drop",          units = r"[-]",          )
        self.HTR_HP_deltaP_des              = Struct(key = "HTR_HP_deltaP_des",                dtype = "float64", repr = "High temp recuperator high pressure design pressure drop",         units = r"[-]",          )
        self.HTR_min_dT                     = Struct(key = "HTR_min_dT",                       dtype = "float64", repr = "High temp recuperator min temperature difference",                 units = r"[C]",          )
        self.HTR_cost_equipment             = Struct(key = "HTR_cost_equipment",               dtype = "float64", repr = "High temp recuperator cost equipment",                             units = r"[M$]",         )
        self.HTR_cost_bare_erected          = Struct(key = "HTR_cost_bare_erected",            dtype = "float64", repr = "High temp recuperator cost equipment and install",                 units = r"[M$]",         )
        self.UA_PHX                         = Struct(key = "UA_PHX",                           dtype = "float64", repr = "PHX Conductance",                                                  units = r"[MW/K]",       )
        self.eff_PHX                        = Struct(key = "eff_PHX",                          dtype = "float64", repr = "PHX effectiveness",                                                units = r"[-]",          )
        self.NTU_PHX                        = Struct(key = "NTU_PHX",                          dtype = "float64", repr = "PHX NTU",                                                          units = r"[-]",          )
        self.T_co2_PHX_in                   = Struct(key = "T_co2_PHX_in",                     dtype = "float64", repr = "CO2 temperature at PHX inlet",                                     units = r"[C]",          )
        self.P_co2_PHX_in                   = Struct(key = "P_co2_PHX_in",                     dtype = "float64", repr = "CO2 pressure at PHX inlet",                                        units = r"[MPa]",        )
        self.deltaT_HTF_PHX                 = Struct(key = "deltaT_HTF_PHX",                   dtype = "float64", repr = "HTF temp difference across PHX",                                   units = r"[C]",          )
        self.q_dot_PHX                      = Struct(key = "q_dot_PHX",                        dtype = "float64", repr = "PHX heat transfer",                                                units = r"[MWt]",        )
        self.PHX_co2_deltaP_des             = Struct(key = "PHX_co2_deltaP_des",               dtype = "float64", repr = "PHX co2 side design pressure drop",                                units = r"[-]",          )
        self.PHX_cost_equipment             = Struct(key = "PHX_cost_equipment",               dtype = "float64", repr = "PHX cost equipment",                                               units = r"[M$]",         )
        self.PHX_cost_bare_erected          = Struct(key = "PHX_cost_bare_erected",            dtype = "float64", repr = "PHX cost equipment and install",                                   units = r"[M$]",         )
        self.PHX_min_dT                     = Struct(key = "PHX_min_dT",                       dtype = "float64", repr = "PHX min temperature difference",                                   units = r"[C]",          )
        self.mc_cooler_T_in                 = Struct(key = "mc_cooler_T_in",                   dtype = "float64", repr = "Low pressure cross flow cooler inlet temperature",                 units = r"[C]",          )
        self.mc_cooler_P_in                 = Struct(key = "mc_cooler_P_in",                   dtype = "float64", repr = "Low pressure cross flow cooler inlet pressure",                    units = r"[MPa]",        )
        self.mc_cooler_rho_in               = Struct(key = "mc_cooler_rho_in",                 dtype = "float64", repr = "Low pressure cross flow cooler inlet density",                     units = r"[kg/$m^{3}$]", )
        self.mc_cooler_m_dot_co2            = Struct(key = "mc_cooler_m_dot_co2",              dtype = "float64", repr = "Low pressure cross flow cooler CO2 mass flow rate",                units = r"[kg/s]",       )
        self.mc_cooler_UA                   = Struct(key = "mc_cooler_UA",                     dtype = "float64", repr = "Low pressure cross flow cooler conductance",                       units = r"[MW/K]",       )
        self.mc_cooler_q_dot                = Struct(key = "mc_cooler_q_dot",                  dtype = "float64", repr = "Low pressure cooler heat transfer",                                units = r"[MWt]",        )
        self.mc_cooler_co2_deltaP_des       = Struct(key = "mc_cooler_co2_deltaP_des",         dtype = "float64", repr = "Low pressure cooler co2 side design pressure drop",                units = r"[-]",          )
        self.mc_cooler_W_dot_fan            = Struct(key = "mc_cooler_W_dot_fan",              dtype = "float64", repr = "Low pressure cooler fan power",                                    units = r"[MWe]",        )
        self.mc_cooler_cost_equipment       = Struct(key = "mc_cooler_cost_equipment",         dtype = "float64", repr = "Low pressure cooler cost equipment",                               units = r"[M$]",         )
        self.mc_cooler_cost_bare_erected    = Struct(key = "mc_cooler_cost_bare_erected",      dtype = "float64", repr = "Low pressure cooler cost equipment and install",                   units = r"[M$]",         )
        self.pc_cooler_T_in                 = Struct(key = "pc_cooler_T_in",                   dtype = "float64", repr = "Intermediate pressure cross flow cooler inlet temperature",        units = r"[C]",          )
        self.pc_cooler_P_in                 = Struct(key = "pc_cooler_P_in",                   dtype = "float64", repr = "Intermediate pressure cross flow cooler inlet pressure",           units = r"[MPa]",        )
        self.pc_cooler_m_dot_co2            = Struct(key = "pc_cooler_m_dot_co2",              dtype = "float64", repr = "Intermediate pressure cross flow cooler CO2 mass flow rate",       units = r"[kg/s]",       )
        self.pc_cooler_UA                   = Struct(key = "pc_cooler_UA",                     dtype = "float64", repr = "Intermediate pressure cross flow cooler conductance",              units = r"[MW/K]",       )
        self.pc_cooler_q_dot                = Struct(key = "pc_cooler_q_dot",                  dtype = "float64", repr = "Intermediate pressure cooler heat transfer",                       units = r"[MWt]",        )
        self.pc_cooler_W_dot_fan            = Struct(key = "pc_cooler_W_dot_fan",              dtype = "float64", repr = "Intermediate pressure cooler fan power",                           units = r"[MWe]",        )
        self.pc_cooler_cost_equipment       = Struct(key = "pc_cooler_cost_equipment",         dtype = "float64", repr = "Intermediate pressure cooler cost equipment",                      units = r"[M$]",         )
        self.pc_cooler_cost_bare_erected    = Struct(key = "pc_cooler_cost_bare_erected",      dtype = "float64", repr = "Intermediate pressure cooler cost equipment and install",          units = r"[M$]",         )
        self.piping_inventory_etc_cost      = Struct(key = "piping_inventory_etc_cost",        dtype = "float64", repr = "Cost of remaining cycle equipment on BEC basis",                   units = r"[M$]",         )
        self.cooler_tot_cost_equipment      = Struct(key = "cooler_tot_cost_equipment",        dtype = "float64", repr = "Total cooler cost equipment",                                      units = r"[M$]",         )
        self.cooler_tot_cost_bare_erected   = Struct(key = "cooler_tot_cost_bare_erected",     dtype = "float64", repr = "Total cooler cost equipment and install",                          units = r"[M$]",         )
        self.cooler_tot_UA                  = Struct(key = "cooler_tot_UA",                    dtype = "float64", repr = "Total cooler conductance",                                         units = r"[MW/K]",       )
        self.cooler_tot_W_dot_fan           = Struct(key = "cooler_tot_W_dot_fan",             dtype = "float64", repr = "Total cooler fan power",                                           units = r"[MWe]",        )
        self.T_state_points                 = Struct(key = "T_state_points",                   dtype = "object",  repr = "Cycle temperature state points",                                   units = r"[C]",          )
        self.P_state_points                 = Struct(key = "P_state_points",                   dtype = "object",  repr = "Cycle pressure state points",                                      units = r"[MPa]",        )
        self.s_state_points                 = Struct(key = "s_state_points",                   dtype = "object",  repr = "Cycle entropy state points",                                       units = r"[kJ/kg-K]",    )
        self.h_state_points                 = Struct(key = "h_state_points",                   dtype = "object",  repr = "Cycle enthalpy state points",                                      units = r"[kJ/kg]",      )
        self.T_LTR_HP_data                  = Struct(key = "T_LTR_HP_data",                    dtype = "object",  repr = "Temperature points along LTR HP stream",                           units = r"[C]",          )
        self.s_LTR_HP_data                  = Struct(key = "s_LTR_HP_data",                    dtype = "object",  repr = "Entropy points along LTR HP stream",                               units = r"[kJ/kg-K]",    )
        self.T_HTR_HP_data                  = Struct(key = "T_HTR_HP_data",                    dtype = "object",  repr = "Temperature points along HTR HP stream",                           units = r"[C]",          )
        self.s_HTR_HP_data                  = Struct(key = "s_HTR_HP_data",                    dtype = "object",  repr = "Entropy points along HTR HP stream",                               units = r"[kJ/kg-K]",    )
        self.T_PHX_data                     = Struct(key = "T_PHX_data",                       dtype = "object",  repr = "Temperature points along PHX stream",                              units = r"[C]",          )
        self.s_PHX_data                     = Struct(key = "s_PHX_data",                       dtype = "object",  repr = "Entropy points along PHX stream",                                  units = r"[kJ/kg-K]",    )
        self.T_HTR_LP_data                  = Struct(key = "T_HTR_LP_data",                    dtype = "object",  repr = "Temperature points along HTR LP stream",                           units = r"[C]",          )
        self.s_HTR_LP_data                  = Struct(key = "s_HTR_LP_data",                    dtype = "object",  repr = "Entropy points along HTR LP stream",                               units = r"[kJ/kg-K]",    )
        self.T_LTR_LP_data                  = Struct(key = "T_LTR_LP_data",                    dtype = "object",  repr = "Temperature points along LTR LP stream",                           units = r"[C]",          )
        self.s_LTR_LP_data                  = Struct(key = "s_LTR_LP_data",                    dtype = "object",  repr = "Entropy points along LTR LP stream",                               units = r"[kJ/kg-K]",    )
        self.T_main_cooler_data             = Struct(key = "T_main_cooler_data",               dtype = "object",  repr = "Temperature points along main cooler stream",                      units = r"[C]",          )
        self.s_main_cooler_data             = Struct(key = "s_main_cooler_data",               dtype = "object",  repr = "Entropy points along main cooler stream",                          units = r"[kJ/kg-K]",    )
        self.T_pre_cooler_data              = Struct(key = "T_pre_cooler_data",                dtype = "object",  repr = "Temperature points along pre cooler stream",                       units = r"[C]",          )
        self.s_pre_cooler_data              = Struct(key = "s_pre_cooler_data",                dtype = "object",  repr = "Entropy points along pre cooler stream",                           units = r"[kJ/kg-K]",    )
        self.P_t_data                       = Struct(key = "P_t_data",                         dtype = "object",  repr = "Pressure points along turbine expansion",                          units = r"[MPa]",        )
        self.h_t_data                       = Struct(key = "h_t_data",                         dtype = "object",  repr = "Enthalpy points along turbine expansion",                          units = r"[kJ/kg]",      )
        self.P_mc_data                      = Struct(key = "P_mc_data",                        dtype = "object",  repr = "Pressure points along main compression",                           units = r"[MPa]",        )
        self.h_mc_data                      = Struct(key = "h_mc_data",                        dtype = "object",  repr = "Enthalpy points along main compression",                           units = r"[kJ/kg]",      )
        self.P_rc_data                      = Struct(key = "P_rc_data",                        dtype = "object",  repr = "Pressure points along re compression",                             units = r"[MPa]",        )
        self.h_rc_data                      = Struct(key = "h_rc_data",                        dtype = "object",  repr = "Enthalpy points along re compression",                             units = r"[kJ/kg]",      )
        self.P_pc_data                      = Struct(key = "P_pc_data",                        dtype = "object",  repr = "Pressure points along pre compression",                            units = r"[MPa]",        )
        self.h_pc_data                      = Struct(key = "h_pc_data",                        dtype = "object",  repr = "Enthalpy points along pre compression",                            units = r"[kJ/kg]",      )
        self.od_rel_tol                     = Struct(key = "od_rel_tol",                       dtype = "int64",   repr = "Off-design Convergence Tolerance",                                 units = r"[-]",          ) 
        self.od_T_t_in_mode                 = Struct(key = "od_T_t_in_mode",                   dtype = "bool",    repr = "Off-design Turbine Mode",                                          units = r"[-]",          ) 
        self.od_opt_objective               = Struct(key = "od_opt_objective",                 dtype = "bool",    repr = "Off-design Optimization Objective",                                units = r"[-]",          ) 
        self.is_gen_od_polynomials          = Struct(key = "is_gen_od_polynomials",            dtype = "bool",    repr = "Off-design Polynomials",                                           units = r"[-]",          ) 
        self.solution_time                  = Struct(key = "solution_time",                    dtype = "float64", repr = "Solve time",                                                       units = r"[s]",          ) 
        self.mc_cooler_in_isen_deltah_to_P_mc_out   = Struct(key = "mc_cooler_in_isen_deltah_to_P_mc_out", dtype = "float64", repr = "Low pressure cross flow cooler inlet isen enthalpy rise to mc outlet pressure",            units = "[kJ/kg]",   )
        self.UA_recup_calc                          = Struct(key = "recup_total_UA_calculated",            dtype = "float64", repr = "Total Recuperator UA",                                                                     units = "[MW/K]",    )
        self.LTR_UA_calculated                      = Struct(key = "LTR_UA_calculated",                    dtype = "float64", repr = "Low temp recuperator UA calculated considering max eff and/or min temp diff parameter",    units = "[MW/K]",    )
        self.HTR_UA_calculated                      = Struct(key = "HTR_UA_calculated",                    dtype = "float64", repr = "High temp recuperator UA calculated considering max eff and/or min temp diff parameter",   units = "[MW/K]",    )
        self.eta_thermal_net_less_cooling_des       = Struct(key = "eta_thermal_net_less_cooling_des",     dtype = "float64", repr = "Calculated cycle thermal efficiency using W_dot_net_less_cooling",                         units = "[-]",       )

class SAMplot(): 
    '''
    Enables users to load data from a source file, apply filters, and
    generate 2D or 3D visualizations with customization. Plots are
    dynamically constructed according to the user's inputs. 

    ### Attributes  
    source : str=''  
        path to the csv data file.  
    dtypes : dict={'header': dtype}  
        (optional) declares the datatype of each column in the csv by header name.   

    ### Plot Options  
    self.x : class Struct  
        defines the x axis  
    self.y : class Struct  
        defines the y axis  
    self.z : class Struct  
        (optional) defines the z axis (contour lines if 2D)  
    self.c : class Struct  
        (optional) defines the color axis  
    self.baseline : tuple=()  
        (optional) plots a single point for reference  
    self.grayscale : False  
        (optional) if ture, converts the plot colors to grayscale  
    self.scatter : False  
        (optional) if true, plots a scatter plot as opposed to a line plot  
    self.legend : False  
        (optional) if true, enables a legend on the plot.   
    self.linelabels : False  
        (optional) if true, attempts to label the z-axis contour lines.  

    ### Methods  
    ```
    build()
        Dynamically loads data and constructs a visual according to all set plot options.
    filter(*args: tuple)
        Rules to exclude data points from display. 
    savefig(path: str, name: str, dpi: float)
        Accesses the protected figure and saves it to file.
    ```
    '''

    def __init__(self, source: str='', dtypes: dict={}):

        # loading the data source
        self.data_full_set = None
        self.resource(source, dtypes)

        # call all pandas preferences here
        pd.options.display.float_format = '{:.2f}'.format

    def resource(self, source: str='', dtypes: dict={}): 
        """
        Loads data from a CSV file into the data_full_set attribute.

        Args:
            source (str): Path to the CSV file to be loaded.
            dtypes (dict): Optional. A dictionary specifying column data types.

        Notes:
            - Uses the 'pyarrow' engine for faster CSV loading.
            - Calls the _defaults() method after loading the data.
        """
        self.source = source
        self.data_full_set = pd.read_csv(self.source, engine='pyarrow', dtype=dtypes)
        self._defaults()

    def normalize(self, params: Parameters = None, param: Struct = None, value: float = None): 
        """
        Normalizes a given parameter and adds it as a new attribute in the params object. 
        Also, it creates a new normalized column in the data set.

        Args:
            params (Parameters): The container object holding the parameter as an attribute.
            param (Struct): The specific parameter to be normalized. Must be an attribute of params.
            value (float): The value by which the parameter's data is normalized.

        Raises:
            ValueError: If the param is not found as an attribute of params.
            KeyError: If param.key is not found in data_full_set.

        Notes:
            - A new attribute is created in params, named as "baseline_<param_name>".
            - The newly created structure's units are set to "[-]" and its repr is prefixed with "Normalized ".
            - A new column is added to data_full_set with the key "norm_<param.key>", 
            containing the normalized data.
        """

        # Helper function to get the attribute name
        def get_attr_name(params, param):
            for name, value in vars(params).items():
                if value is param:
                    return name
            raise ValueError("Parameter not found as an attribute of the given object.")

        # Creating the normalized parameter
        name = f"{get_attr_name(params, param)}_norm"
        structure = Struct(
            key = "norm_" + param.key,
            repr = "Normalized " + param.repr,
            units = "[-]",
            dtype = param.dtype,
            acro = param.acro+r"$_{norm}$",
            default = param.default,
            value = value
        )

        setattr(params, name, structure)

        # Normalizing the data and adding it to the data set
        if structure.key not in self.data_full_set.columns:
            if param.key in self.data_full_set.columns:
                # Perform the normalization and concatenate with the existing DataFrame
                self.data_full_set = pd.concat([self.data_full_set, 
                                                pd.Series(self.data_full_set[param.key] / value, 
                                                        name=structure.key)], axis=1)
            else:
                raise KeyError(f"'{param.key}' not found in data_full_set.")
        else:
            print(f"'{structure.key}' already exists in data_full_set.")

        # Optionally, make a copy of the DataFrame to reduce fragmentation
        self.data_full_set = self.data_full_set.copy()

    def show(self): 
        '''
        Displays the visual constructed using .build(). 
        '''
        if self._message: print(self._message)
        if self._plot_case != 0: self.fig.tight_layout()
        plt.show()

    def save(self, path: str=os.getcwd(), name: str='figure', dpi: float=300): 
        '''
        Saves the visual to the current working directory or the path provided.

        ### Parameters
        ---
        path : str=os.getcwd()
            (optional) Defines the folder to save the visual in.
        name : str='figure'
            (optional) Defines the save name of the visual. Extension not required. 
        dpi : float=300
            (optional) Defines the dots per inch the visual is saved with. 
        '''

        self.fig.savefig(os.path.join(path, name + '.png'), dpi=dpi, bbox_inches='tight')

    def filter(self, *args: tuple):
        '''
        Dynamically excludes data points from display according to a Struct
        parameter and a lambda function evaluating to True or False for
        each item in the column.

        Example
        ```
        samplt = Plotter()
        params = Parameters()

        samplt.filter(
            (params.try_s_cycle, lambda x: x == 0), 
            (params.PHX_hot_in, lambda x: x >= 800), 
            etc.
        )
        ``` 

        ### Parameters
        arg : tupel
            
            First element is of the type Struct, second element is a lambda function. 
            Optionally, the second element may be of the form ({min, max}, Struct). 
        '''

        self._filters = []
        for column, condition in args: 
            if isinstance(column, Struct): 

                if isinstance(condition, types.LambdaType): 
                    self._filters.append(lambda df, col=column, cond=condition: df[col.key].map(cond))
                elif isinstance(condition, tuple) and condition[0]==min and isinstance(condition[1], Struct): 
                    self._filters.append(
                        lambda df, xcol=column, ycol=condition[1]: 
                            df[ycol.key] == df[xcol.key].map(
                                df.groupby(xcol.key)[ycol.key].min()
                            )
                        )
                elif isinstance(condition, tuple) and condition[0]==max and isinstance(condition[1], Struct): 
                    self._filters.append(
                        lambda df, xcol=column, ycol=condition[1]: 
                            df[ycol.key] == df[xcol.key].map(
                                df.groupby(xcol.key)[ycol.key].max()
                            )
                        )
                else: raise TypeError('Conditions must be given as a lambda function or as ({min, max}, Struct)).')

            ### NOT WORKING AS INTENDED. NEEDS DEBUGGING. 
            # - OR operator doesn't seem to apply the filter to each test value exclusively
            elif isinstance(column, tuple) and isinstance(column[0], Struct) and isinstance(column[1], Struct): 

                contours = np.sort(self.data_full_set[column[1].key].dropna().unique())
                if isinstance(condition, tuple) and condition[0]==min and isinstance(condition[1], Struct):
                    for test in contours: 
                        self._filters.append(
                            lambda df, xcol=column[0], ycol=condition[1], zcol=column[1], test=test: 
                                (df[zcol.key] != test) | (df[ycol.key] == df[xcol.key].map(
                                    df.groupby(xcol.key)[ycol.key].min()
                                ))
                            )
                elif isinstance(condition, tuple) and condition[0]==max and isinstance(condition[1], Struct):
                    for test in contours: 
                        self._filters.append(
                            lambda df, xcol=column[0], ycol=condition[1], zcol=column[1], test=test: 
                                (df[zcol.key] != test) | (df[ycol.key] == df[xcol.key].map(
                                    df.groupby(xcol.key)[ycol.key].max()
                                ))
                            )
                else: raise TypeError('Conditions must be given as a lambda function or as ({min, max}, Struct)).')
            else: raise TypeError('Filters must be of the form (Struct, lambda), (Struct, ({max/min}, Struct), or ((Struct, Struct), ({min/max}, Struct)))')

        try: self._last_set = self.data
        except: pass

        self.data = self.data_full_set
        for f in self._filters: 
            self.data = self.data[f(self.data)]

    def build(self, title: str='', label: str='', style: str=''): 
        '''
        Dynamically loads data and constructs a visual according to 
        all set plot options.

        ### Parameters
        ---
        title : str=''
            (optional) Defines the displayed title of the visual. 
        '''

        self._label = label
        self._title = title
        self._line_style = style
        self._plot_case = (bool(self.x) * self._x) + (bool(self.y) * self._y) + (bool(self.z) * self._z) + (bool(self.c) * self._c) + (bool(self.plot3d) * self._D)
        match self._plot_case: 
            case  0: # no axis, describes the optimal cycle
                self._build_pie()
            case  3: # x+y, standard plot
                self._build_xy()
            case  7: # x+y+z, standard plot with shaded contours
                self._build_xyz()
            case 15: # x+y+z+c, shaded contours and colorbar
                self._build_xyzc()
            case 23: # x+y+z, monochrome 3d plot
                self._build_xyz_3d()
            case 31: # x+y+z, 3d plot with shaded contours and colorbar
                self._build_xyzc_3d()
            case _: 
                raise AttributeError('Plot build configuration not available.')

        if self.legend: plt.legend()

    def reset(self): 
        '''
        Clears the current figure, axis, and plot options. 
        '''
        self._garbage = self.data_full_set
        del self._garbage
        gc.collect()
        plt.close()

    def _getdata(self, *args): 
        
        if not self._filters: 
            try: self._last_set = self.data 
            except: pass

            self.data = self.data_full_set

        try: # Validate args structure and filter columns
            keys = [arg.key for arg in args]
            if keys: self.data = self.data[keys]
            if self.x: self.data = self.data.sort_values(by=self.x.key, ascending=True, ignore_index=True)

            if self.data.empty: 
                raise ValueError("Data set is empty and cannot be processed.")

        except KeyError as e:
            raise ValueError(f"Invalid key structure in arguments: {e}")
        except ValueError as e: 
            raise ValueError(f"{e}")

    def _build_base(self): 

        if self.baseline: 

            self.ax.scatter(
                self.baseline[0], 
                self.baseline[1], 
                label='Baseline', 
                color='black', 
                zorder=3, s=12
            )

            self.ax.legend()
        
        if not self.scatter: 
            self.ax.margins(x=0, y=0.05)
            if not self._line_style: 
                self._line_style = next(self._line_styles)
            else: pass
        self.ax.set_xlabel(f"{self.x.repr} {self.x.units}")
        self.ax.set_ylabel(f"{self.y.repr} {self.y.units}")
        self.ax.set_title(self._title)
        self.ax.grid(True)

    def _build_base_3d(self): 
        self.fig.clf()
        self.fig.set_size_inches(7, 5)
        self.ax = self.fig.add_subplot(111, projection='3d')

        if self.baseline: 

            self.ax.scatter(
                self.baseline[0], 
                self.baseline[1], 
                self.baseline[2], 
                label='Baseline', 
                color='black', 
                zorder=3, s=12
            )

            self.ax.legend()
        
        self._fontsize = 8
        self.ax.set_xlabel(f"{self.x.repr} {self.x.units}", fontsize=self._fontsize)
        self.ax.set_ylabel(f"{self.y.repr} {self.y.units}", fontsize=self._fontsize)
        self.ax.set_zlabel(f"{self.z.repr} {self.z.units}", fontsize=self._fontsize)
        self.ax.tick_params(axis='x', labelsize=self._fontsize) 
        self.ax.tick_params(axis='y', labelsize=self._fontsize) 
        self.ax.tick_params(axis='z', labelsize=self._fontsize) 

        self.ax.set_title(self._title)
        self.ax.grid(True)

    def _build_base_pie(self): 
        plt.close()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(9, 5))
        self.fig.subplots_adjust(wspace=-0.3)

    def _build_pie(self):
        self._getdata()
        self._build_base_pie()

        #---setting up data structures for total system and power cycle
        params = Parameters()
        series = self.data.loc[
            self.data[params.levelized_cost_of_energy.key].idxmin()
            ].copy()
        self.data = series

        plant = [
            params.cycle_capital_cost,
            params.solar_tower_cost, 
            params.solar_field_cost, 
            params.receiver_cost, 
            params.particles_cost, 
            params.particle_storage_cost, 
            params.particle_lifts_cost, 
            params.land_cost, 
            params.balance_of_plant_cost, 
        ]

        cycle = [
            params.HTR_capital_cost, 
            params.LTR_capital_cost, 
            params.PHX_capital_cost, 
            params.air_cooler_capital_cost, 
            params.compressor_capital_cost, 
            params.recompressor_capital_cost, 
            params.turbine_capital_cost, 
            params.piping_capital_cost
        ]

        #---setting up chart parameters
        cycle_cost = series[params.cycle_capital_cost.key]
        plant_cost = series[params.plant_capital_cost.key]
        total_cost = series[params.total_capital_cost.key]

        cycle_values = [series[eq.key] for eq in cycle]
        cycle_ratios = [series[eq.key] / cycle_cost for eq in cycle]
        cycle_labels = [eq.acro for eq in cycle]

        plant_values = [series[eq.key] for eq in plant]
        plant_ratios = [series[eq.key] / total_cost for eq in plant]
        plant_labels = [eq.acro for eq in plant]

        # aggregating plotted values if less than 1% of total cost
        other_ratio = 0.0
        other_value = 0.0
        all_removed = []
        for i, ratio in enumerate(plant_ratios):
            if ratio <= 0.03: 
                other_value += plant_values.pop(i)
                other_ratio += plant_ratios.pop(i)
                all_removed.append(plant_labels.pop(i))
            else: pass
        for i, ratio in enumerate(cycle_ratios):
            if ratio <= 0.004: 
                other_value += cycle_values.pop(i)
                other_ratio += cycle_ratios.pop(i)
                all_removed.append(cycle_labels.pop(i))
            else: pass
        
        plant_values.append(other_value)
        plant_ratios.append(other_ratio)
        plant_labels.append("Other")

        #---creating the pie chart
        fontsize = 6

        explode = [0.05 if eq == params.cycle_capital_cost.acro else 0.0 for eq in plant_labels]
        colors = sns.color_palette("crest", len(plant_ratios))
        angle = -180 * plant_ratios[0]
        wedge, *_ = self.ax1.pie(
            plant_ratios, 
            autopct = '%1.1f%%', 
            labels = plant_labels, 
            startangle = angle, 
            explode = explode, 
            pctdistance=0.85, 
            colors=colors
        )

        for text in self.ax1.texts: 
            text.set_fontsize(fontsize)

        #---creating the bar plot
        bottom = 1.0
        factor = 4.0
        width = 0.1

        # Adding from the top matches the legend.
        for j, (height, label) in enumerate(reversed([*zip(cycle_ratios, cycle_labels)])):
            bottom -= height
            bc = self.ax2.bar(
                0, height, width, bottom=bottom, color=colors[0], 
                label=label, alpha=0.1 + (j / len(cycle_ratios))
            )
            self.ax2.bar_label(bc, labels=[f"{height:.1%}"], label_type='center', fontsize=fontsize)

            for rect in bc:
                self.ax2.text(
                    0.02 + rect.get_x() + rect.get_width(), 
                    rect.get_y() + rect.get_height() / 2, 
                    label, 
                    va='center', 
                    ha='left', 
                    fontsize=fontsize 
                )

        self.ax2.set_title('')
        self.ax2.axis('off')
        self.ax2.set_xlim(-factor * width, factor * width)
        self.ax2.set_ylim(-0.2, 1.2)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedge[0].theta1, wedge[0].theta2
        center, r = wedge[0].center, wedge[0].r
        bar_height = sum(cycle_ratios)

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=self.ax2.transData,
                            xyB=(x, y), coordsB=self.ax1.transData)
        con.set_color('gray')
        con.set_linewidth(0.2)
        self.ax2.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=self.ax2.transData,
                            xyB=(x, y), coordsB=self.ax1.transData)
        con.set_color('gray')
        self.ax2.add_artist(con)
        con.set_linewidth(0.2)

        self._message = (
            f"""
#---Best Design, Cycle Summary
{params.eta_thermal_calc.repr:.<30}{series[params.eta_thermal_calc.key]:.>10.3f} {params.eta_thermal_calc.units}
{params.PHX_hot_in.repr:.<30}{series[params.PHX_hot_in.key]:.>10.2f} {params.PHX_hot_in.units}
{params.PHX_dT_hot.repr:.<30}{series[params.PHX_dT_hot.key]:.>10.2f} {params.PHX_dT_hot.units}
{params.PHX_dT_cold.repr:.<30}{series[params.PHX_dT_cold.key]:.>10.2f} {params.PHX_dT_cold.units}
{params.q_dot_PHX.repr:.<30}{series[params.q_dot_PHX.key]:.>10.2f} {params.q_dot_PHX.units}
{params.T_co2_PHX_in.repr:.<30}{series[params.T_co2_PHX_in.key]:.>10.2f} {params.T_co2_PHX_in.units}

{params.cycle_capital_cost.repr:.<30}{series[params.cycle_capital_cost.key]:.>10.2f} {params.cycle_capital_cost.units}
{params.plant_capital_cost.repr:.<30}{series[params.plant_capital_cost.key]:.>10.2f} {params.plant_capital_cost.units}
{params.total_adjusted_cost.repr:.<30}{series[params.total_adjusted_cost.key]:.>10.2f} {params.total_adjusted_cost.units}
{params.levelized_cost_of_energy.repr:.<30}{series[params.levelized_cost_of_energy.key]:.>10.2f} {params.levelized_cost_of_energy.units}
            """
        )

    def _build_xy(self): 
        self._getdata(self.x, self.y)
        self._build_base()

        if not self.grayscale: 
            color = '#221330'
        else: color = 'black'

        if not self.scatter: 
            self.ax.plot(self.data[self.x.key], self.data[self.y.key], color=color, linestyle=self._line_style, label=self._label)
        else: self.ax.scatter(self.data[self.x.key], self.data[self.y.key], color=color, zorder=3)

    def _build_xyz(self): 
        self._getdata(self.x, self.y, self.z)
        self._build_base()

        # tolerance = 0.1
        # contours = np.sort(np.unique(np.round(self.data[self.z.key] / tolerance) * tolerance))
        contours = np.sort(self.data[self.z.key].dropna().unique())
        if not self.grayscale: 
            colors = sns.color_palette('rocket', len(contours))
            alpha = 0.0
            cmap = sns.color_palette('rocket', as_cmap = True)
        else: 
            colors = sns.color_palette('gray', len(contours))
            alpha = 0.3
            cmap = plt.get_cmap('gray')

        if self.grayscale: 

            for i, z in enumerate(contours): 
                subset = self.data[self.data[self.z.key] == z]
                if not self.scatter: 
                    self.ax.plot(subset[self.x.key], subset[self.y.key], label=f"{z}", color='black', alpha=alpha)
                else: self.ax.scatter(subset[self.x.key], subset[self.y.key], color='black', alpha=alpha, zorder=2)
            if self.linelabels: labelLines(self.ax.get_lines(), align=True, zorder=2.5, fontsize=6)
            
        for i, z in enumerate(contours): 
            subset = self.data[self.data[self.z.key] == z]
        # for i, z in enumerate(contours): 
        #     subset = self.data[np.round(self.data[self.z.key] / tolerance) * tolerance == z]

            if not self.scatter: 
                self.ax.plot(subset[self.x.key], subset[self.y.key], label=f"{z}", color=colors[i], linestyle=self._line_style)
            else: self.ax.scatter(subset[self.x.key], subset[self.y.key], color=colors[i], zorder=2)
        if self.linelabels and not self.grayscale: labelLines(self.ax.get_lines(), align=True, zorder=2.5, fontsize=6)

        norm = mcolors.Normalize(vmin=min(contours), vmax=max(contours))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax)
        cbar.set_label(f"{self.z.repr} {self.z.units}")

    def _build_xyzc(self): 
        self._getdata(self.x, self.y, self.z, self.c)
        self._build_base()

        contours = self.data[self.z.key].dropna().unique()
        if not self.grayscale: 
            colors = sns.color_palette('rocket', len(contours))
            alpha = 0.0
            cmap = sns.color_palette('rocket', as_cmap = True)
        else: 
            colors = sns.color_palette('gray', len(contours))
            alpha = 0.3
            cmap = plt.get_cmap('gray')

        if self.linelabels: 
            # generating each plot
            for i, z in enumerate(contours): 
                subset = self.data[self.data[self.z.key] == z]
                self.ax.plot(subset[self.x.key], subset[self.y.key], label=f'{z}', color='black', alpha=alpha)
            
            labelLines(self.ax.get_lines(), align=True, zorder=2.5, fontsize=6)

        # generating the color map and colorbar
        mapval = self.data[self.c.key].dropna()
        if not self.grayscale: cmap = sns.color_palette('rocket', as_cmap = True)
        else: cmap = plt.get_cmap('gray')
        norm = mcolors.Normalize(vmin=min(mapval), vmax=max(mapval))

        # generating each plot
        for i, z in enumerate(contours): 
            subset = self.data[self.data[self.z.key] == z]

            segs, colors, data = self._lineColorGradient(
                xs=subset[self.x.key].to_numpy(), 
                ys=subset[self.y.key].to_numpy(), 
                metric=subset[self.c.key].to_numpy(), 
                cmap=cmap
            )

            if self.scatter: # plotting points as well as line segments
                self.ax.scatter(data[0][:-1], data[1][:-1], marker='.', color=colors, zorder=3.0)
                self.ax.scatter(data[0][-1], data[1][-1], marker='.', color=colors[-1], zorder=3.0)
            self.ax.add_collection(LineCollection(segs, colors=colors))

        # plotting options and handling
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax)
        cbar.set_label(self.c.repr)
        if self.legend: self.ax.legend()

    def _build_xyz_3d(self): 
        self._getdata(self.x, self.y, self.z)
        self._build_base_3d()

        if not self.grayscale: 
            color = '#221330'
            cmap = sns.color_palette('rocket', as_cmap = True)
        else: 
            color = 'black'
            cmap = plt.get_cmap('gray')
        alpha = 0.5

        try: 
            if not self.scatter: 
                trisurf = tri(self.data[self.x.key], self.data[self.y.key])
                self.ax.plot_trisurf(self.data[self.x.key], self.data[self.y.key], self.data[self.z.key], triangles=trisurf.triangles, cmap=cmap, edgecolors='none')
            else: self.ax.scatter(self.data[self.x.key], self.data[self.y.key], self.data[self.z.key], color=color, s=50, alpha=alpha)
        except: pass

    def _build_xyzc_3d(self): 
        self._getdata(self.x, self.y, self.z, self.c)
        self._build_base_3d()

        self.data = pd.concat([self._last_set, self.data])

        alpha = 0.8
        if not self.grayscale: 
            cmap = sns.color_palette('rocket', as_cmap = True)
        else: cmap = plt.get_cmap('gray')

        try: 
            if not self.scatter: 
                trisurf = tri(self.data[self.x.key], self.data[self.y.key])
                self.ax.plot_trisurf(self.data[self.x.key], self.data[self.y.key], self.data[self.z.key], triangles=trisurf.triangles, cmap=cmap, edgecolors='none')
            else: self.ax.scatter(self.data[self.x.key], self.data[self.y.key], self.data[self.z.key], c=self.data[self.c.key], cmap=cmap, s=50, alpha=alpha)
        except: pass

        ## HOT FIX - CBAR NOT WORKING WITH tri surface plot
        if self.scatter: 
            norm = mcolors.Normalize(vmin=min(self.data[self.c.key]), vmax=max(self.data[self.c.key]))
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, shrink=0.8, pad=0.12)
            cbar.set_label(f"{self.c.repr} {self.c.units}", fontsize=self._fontsize, labelpad=10)
            cbar.ax.tick_params(labelsize=self._fontsize)

    def _defaults(self): 
        self.x = False  # primary axis
        self.y = False  # secondary axis
        self.z = False  # multiple lines / tertiary axis
        self.c = False  # colorbar / line gradient axis

        # general class members
        self._message = None

        # axis values to match-case the correct print function
        self._x = 0b00001
        self._y = 0b00010
        self._z = 0b00100
        self._c = 0b01000
        self._D = 0b10000 

        # initializing optional data structures
        self._last_set = pd.DataFrame()
        self._filters = []
        self.baseline = ()

        # initializing plotter optionals
        self.linelabels = False
        self.grayscale = False
        self.scatter = False
        self.legend = False
        self.plot3d = False

        # the figure and axis is created here so they get reset after .show()
        self.fig, self.ax = plt.subplots(figsize=(5.5, 4), dpi=100)
        self._line_styles = self._lineStyleGenerator()

    def _lineColorGradient(self, xs, ys, metric=None, cmap=plt.cm.viridis, n_resample=20):
        '''
        Generates a set of line segments and matching colors
        for creating a line-gradient plot.  
        '''
        n_points = len(xs)
        xsInterp = np.linspace(0, 1, n_resample)
        segments = []
        segmentColors = []
        xpts = [xs[0]]
        ypts = [ys[0]]

        # Normalize zs to [0, 1]
        zmin, zmax = min(metric), max(metric)
        norm = mcolors.Normalize(vmin=zmin, vmax=zmax)
        
        for i in range(n_points - 1):
            # Calculate the midpoint for interpolation
            xfit = interp([0, 1], xs[i:i+2])
            yfit = interp([0, 1], ys[i:i+2])
            xmid = xfit(xsInterp)
            ymid = yfit(xsInterp)
            xpts.extend(xmid[1:])
            ypts.extend(ymid[1:])

            # Calculate the color for each segment based on zs
            zfit = interp([0, 1], metric[i:i+2]) 
            zmid = zfit(xsInterp)
            segmentColors.extend([cmap(norm(z))[:3] for z in zmid[1:]])  # exclude alpha

            # Initialize the start and close points for each segment
            start = [xmid[0], ymid[0]]
            close = [xmid[0], ymid[0]]
            
            # Finalize each segment
            for x, y in zip(xmid[1:], ymid[1:]):
                start = [close[0], close[1]]
                close = [x, y]
                segments.append([start, close])

        # Combine the colors with an alpha of 1 for each segment
        colors = [(*color, 1) for color in segmentColors]
        data = [xpts, ypts]
        return segments, colors, data

    def _lineStyleGenerator(self):
        """
        Initialize the line style generator.
        """
        
        styles = [
            '-', '--', '-.', ':', 
            (0, (3, 1)), (0, (5, 2)),
            (0, (1, 2, 5, 2)), (0, (4, 1, 2, 1)), (0, (3, 1, 3, 1))
        ]
        
        return itertools.cycle(styles)

if __name__=='__main__': 

    # source = os.path.join(os.getcwd(), 'SSC CSP API', 'results', '2025-02-10_full solution.csv')
    # source = os.path.join(os.getcwd(), 'SSC CSP API', 'results', '2025-02-14_cost basis sensitivity.csv')
    # source = os.path.join(os.getcwd(), 'SSC CSP API', 'results', '2025-02-18_heliostat cost sensitivity.csv')
    # source = os.path.join(os.getcwd(), 'SSC CSP API', 'results', '2025-02-18_receiver eta sensitivity.csv')
    source = os.path.join(os.getcwd(), 'SSC CSP API', 'results', '2025-02-28_full solution.csv')

    params = Parameters()
    dtypes = {par.key: par.dtype for par in params.get()}
    samplt = SAMplot(source, dtypes=dtypes)
    samplt.normalize(params, params.levelized_cost_of_energy, 65.50304439)

    samplt.x = params.PHX_cost_basis
    samplt.y = params.m_dot_htf_des
    samplt.z = params.PHX_hot_in

    samplt.legend = False
    samplt.plot3d = False
    samplt.scatter = True
    samplt.grayscale = False
    samplt.linelabels = False

    samplt.filter(
        (params.try_s_cycle, lambda x: x == 1), 
        (params.UA_recup_tot, lambda x: x != 30000), 
        (params.PHX_dT_cold, lambda x: x == 20), 
        (params.rec_eta_mod, lambda x: x >= 0.99), 
        # (params.PHX_dT_hot, lambda x: x == 240), 
        # (params.PHX_hot_in, lambda x: x <= 950), 
        # (params.PHX_cost_basis, lambda x: x == 100), 
        # (params.levelized_cost_of_energy, lambda x: x <= 60), 
        (params.PHX_cost_basis, (min, params.levelized_cost_of_energy)), 
    )

    samplt.build()
    samplt.show()
    quit()

    def phx_design_space(): 

        samplt.x = params.PHX_cost_basis
        samplt.y = params.levelized_cost_of_energy_norm
        samplt.z = params.PHX_hot_in

        samplt.legend = False
        samplt.plot3d = False
        samplt.scatter = False
        samplt.grayscale = False
        samplt.linelabels = False

        samplt.filter(
            (params.try_s_cycle, lambda x: x == 1), 
            (params.UA_recup_tot, lambda x: x != 30000), 
            (params.PHX_dT_cold, lambda x: x == 20), 
            (params.PHX_dT_hot, lambda x: x == 280), 
            (params.PHX_hot_in, lambda x: x % 20 == 0 and x <= 1100 and x >= 760 and x not in [980]), 
            (params.rec_eta_mod, lambda x: x >= 0.99), 
            (params.PHX_cost_basis, (min, params.levelized_cost_of_energy))
        )

        samplt.build()
        for line in plt.gca().get_lines():
            line.set_label('')
        for line in samplt.ax.lines:
            line.remove()

        samplt.ax.scatter(100, 1, marker='D', label='baseline', color='black', zorder=3, s=12)

        samplt.ax.plot(
            [50, samplt.data[params.PHX_cost_basis.key].max()], 
            [1.0, 1.0], 
            color = 'black', 
            linestyle = '--', 
            # linewidth = 1.0
        )

        samplt.ax.plot(
            [50, samplt.data[params.PHX_cost_basis.key].max()], 
            [0.985, 0.985], 
            color = 'gray', 
            linestyle = '-', 
            linewidth = 1.0, 
            zorder = 0.5, 
            alpha = 0.3, 
        )

        samplt.ax.set_ylim(top=1.03)

        samplt.filter(
            (params.try_s_cycle, lambda x: x == 1), 
            (params.UA_recup_tot, lambda x: x != 30000), 
            (params.PHX_dT_cold, lambda x: x == 20), 
            (params.rec_eta_mod, lambda x: x >= 0.99), 
            (params.levelized_cost_of_energy, lambda x: x <= 70), 
            (params.PHX_cost_basis, (min, params.levelized_cost_of_energy)), 
        )

        error = 0.0
        # $13365/kg to print, assume 100x decrease in production
        # divide by $/UA of FP, 316H PHX, assuming materials = 30% of total cost
        SiC_cost_basis = 240
        SiC_LCOE_opted = np.interp(
            SiC_cost_basis, 
            samplt.data[params.PHX_cost_basis.key], 
            samplt.data[params.levelized_cost_of_energy_norm.key]
        )

        samplt.ax.errorbar(
            SiC_cost_basis, SiC_LCOE_opted, 
            xerr=SiC_cost_basis * error, 
            fmt='o', 
            color='black', 
            capsize=0, 
            markersize=4, 
            linewidth=0, 
            capthick=0, 
            label='TO SiC'
        )

        samplt.x = params.PHX_cost_basis
        samplt.y = params.levelized_cost_of_energy_norm
        samplt.z = None

        samplt.legend = True
        samplt.plot3d = False
        samplt.scatter = False
        samplt.grayscale = False
        samplt.linelabels = False

        samplt.build(style='--')
        samplt.save(name='PHX Design Space')
        samplt.show()

    phx_design_space()

    # samplt.save(name='PIT-LCOE-dTh')

    # samplt.baseline = (100, 65.1954)
    # ---
    # ~55% decrease in UA cost basis for 1 $/MWh reduction @ 700C
    # ~10C increase in PIT for 1 $/MWh reduction in LCOE @ 700C
    # best case for same cost basis: 52.46 $/MWh (8.735 $/MWh reduction in LCOE)
    # very high cost bases still meet LCOE reduction requirement

    # for getting PHX temperatures
    # ---
    # print(f"{params.T_turb_in.repr:.<50}{samplt.data[params.T_turb_in.key]:.>8.2f} {params.T_turb_in.units}")
    # print(f"{params.T_co2_PHX_in.repr:.<50}{samplt.data[params.T_co2_PHX_in.key]:.>8.2f} {params.T_co2_PHX_in.units}")
    # print(f"{'HTF Inlet Temperature (PHX Inlet)':.<50}{samplt.data[params.T_htf_cold_des.key]+samplt.data[params.deltaT_HTF_PHX.key]:.>8.2f} {params.T_htf_cold_des.units}")
    # print(f"{params.T_htf_cold_des.repr:.<50}{samplt.data[params.T_htf_cold_des.key]:.>8.2f} {params.T_htf_cold_des.units}")


