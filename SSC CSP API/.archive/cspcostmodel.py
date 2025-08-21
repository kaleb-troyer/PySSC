
import matplotlib.pyplot as plt
import utilities as ut
import seaborn as sns
import pandas as pd
import numpy as np

class costModel():
    '''
    This cost model is intended for use as a pythonic analog to the
    total system cost model implemented in SAM Simulation Core. The
    cost model assumes a static power block size and cost, and as a
    result, should only be used to analyze the relationship between
    power block thermal efficiency and the costs of a g3 CSP plant.

    Otherwise, the cost model is useful for analyzing changes to 
    any of the CSP cost functions before implementation into SSC. 
    
    model = costModel()
    -> instantiates the cost model
    model.analysis(eta_thermal=0.5)
    -> calculates costs for a given power block efficiency
    model.display()
    -> displays the results of an analysis
    '''

    def __init__(self) -> None:
        self.costLND = 0
        self.costTWR = 0
        self.costREC = 0
        self.costLFT = 0
        self.costHTF = 0
        self.costTES = 0
        self.costFLD = 0
        self.costCYC = 0
        self.costTOT = 0        
    def display(self) -> None: 
        print(f"")
        print(f"W_dot_net = {self.W_dot_net:7.2f} [MWe]")
        print(f"m_dot_phx = {self.m_dot_hot_des:7.2f} [kg/s]")
        print(f"-----------------------------------------")
        print(f"{'costLND':.<18}{self.costLND:.>10.2f} [M$]") 
        print(f"{'costTWR':.<18}{self.costTWR:.>10.2f} [M$]") 
        print(f"{'costREC':.<18}{self.costREC:.>10.2f} [M$]") 
        print(f"{'costLFT':.<18}{self.costLFT:.>10.2f} [M$]") 
        print(f"{'costHTF':.<18}{self.costHTF:.>10.2f} [M$]") 
        print(f"{'costTES':.<18}{self.costTES:.>10.2f} [M$]") 
        print(f"{'costFLD':.<18}{self.costFLD:.>10.2f} [M$]") 
        print(f"{'costCYC':.<18}{self.costCYC:.>10.2f} [M$]") 
        print(f"") 
        print(f"{'Total capital':.<18}{self.costTOT:.>10.2f} [M$]")
        print(f"{'Calculated LCOE':.<18}{self.LCOE:.>10.2f} [$/MW-h]")
        print(f"")
    def analysis(self, eta_thermal : float=0.5) -> dict: 

        # determine these somehow
        self.deltaT          = 20
        self.T_phx_i         = 700 + 273.15 + self.deltaT
        self.T_phx_o         = 470 + 273.15 + self.deltaT
        self.t_averaged      = (self.T_phx_o + self.T_phx_i) / (2 * 1000)

        self.solar_multiple  = 2.5
        self.eta_receiver    = 0.9183
        # power requirements
        self.eta_thermal     = eta_thermal
        self.W_dot_net       = 100 # [MWe]
        self.W_dot_thm       = self.W_dot_net / self.eta_thermal
        self.W_dot_field     = self.solar_multiple * self.W_dot_thm / self.eta_receiver
        self.W_dot_htf       = self.solar_multiple * self.W_dot_thm 

        # CSP sizing
        self.A_receiver      = ( 9.113e+00 + 3.274e-02 * self.W_dot_field) * 15
        self.H_tower         = ( 4.821e+01 + 4.447e-01 * self.W_dot_field)
        self.A_field         = (-6.272e+04 + 2.174e+03 * self.W_dot_field)

        # particle properties
        self.rho             = 1625.0
        self.cp              = 0.1850
        self.angle           = 0.5590
        self.heat_capacity   = (1 / (1000 * 0.0600843)) * (-6.076591 + 251.6755 * self.t_averaged + 
            (-324.7964) * self.t_averaged**2 + 168.5604 * self.t_averaged**3 + 0.002548 / self.t_averaged**2)
        self.m_dot_hot_des   = 1e6 * self.W_dot_thm / (1e3 * self.heat_capacity * (self.T_phx_i - self.T_phx_o)) * 1.58

        # thermal energy storage bin
        self.hours           = 14
        self.r_bin           = 12
        self.m_htf           = self.m_dot_hot_des * self.hours * 3600
        self.V_htf           = self.m_htf / self.rho
        self.H_bin           = (self.V_htf - ((np.pi / 3) * pow(self.r_bin, 3) * np.tan(self.angle))) / (np.pi * pow(self.r_bin, 2))
        self.A_bin_surf      = 2 * np.pi * self.r_bin * self.H_bin + np.pi * self.r_bin * np.sqrt(self.H_bin * self.H_bin + self.r_bin * self.r_bin)

        # LCOE parameters
        self.total_life      = 30.00
        self.capacity_factor = 0.700
        self.o_and_m         = 40000
        self.f_cont          = 0.100
        self.f_cons          = 0.060
        self.f_ind           = 0.130
        self.f_fin           = 0.070
        self.i_inf           = 0.025
        self.f_prime         = ((1 + self.f_fin) / (1 + self.i_inf)) - 1
        self.CRF             = self.f_prime * pow(1 + self.f_prime, self.total_life) / (pow(1 + self.f_prime, self.total_life) - 1)

        # other parameters
        self.eta_lft         = 0.8
        self.m_dot_p         = self.m_dot_hot_des * self.solar_multiple
        self.H_lifts         = self.H_bin * 3
        self.c_bin_h         = 1230 + 0.37 * ((self.T_phx_i - 600) / 400) 
        self.c_bin_c         = 1230 + 0.37 * ((self.T_phx_o - 600) / 400) 
        self.f_losses        = 1e-6
        self.c_losses        = self.total_life * self.cp * self.m_dot_p * (self.hours / self.solar_multiple) * 365 * self.f_losses
        self.non_storage     = 0.05

        # power parasitics
        self.W_dot_lift      = 1e-6 * self.m_dot_p * self.H_lifts * 9.80665 / self.eta_lft
        self.W_dot_cool      = 2.0
        self.W_dot_helio     = self.W_dot_htf * 0.0055
        self.W_dot_less      = self.W_dot_net - self.W_dot_lift - self.W_dot_cool - self.W_dot_helio

        # LCOE calculation 
        self.W_annual        = self.capacity_factor * self.W_dot_less * 24 * 365
        self.installed_cost  = (1 + self.f_cons) * (1 + self.f_ind) * ((1 + self.f_cont) * self._totalCost()) * 1E6

        if abs(self.eta_thermal - 0.442105) < 1e-6: 
            print(self.m_dot_p)
            print(self.H_bin)
            print(self.H_tower)
            quit()

        self.LCOE            = ((self.installed_cost * self.CRF) + (self.o_and_m * self.W_dot_net)) / self.W_annual

        analysis_dict = {
            "eta_cyc": self.eta_thermal, 
            "costLND": self.costLND, 
            "costTWR": self.costTWR, 
            "costREC": self.costREC, 
            "costLFT": self.costLFT, 
            "costHTF": self.costHTF, 
            "costTES": self.costTES, 
            "costFLD": self.costFLD, 
            "costCYC": self.costCYC,  
            "costCSP": self.costCSP,
            "costTOT": self.costTOT,
            "totLCOE": self.LCOE, 
        }

        return analysis_dict

    def _totalCost(self): 

        self.costLND = self._costLND()
        self.costTWR = self._costTWR()
        self.costREC = self._costREC()
        self.costLFT = self._costLFT()
        self.costHTF = self._costHTF()
        self.costTES = self._costTES()
        self.costFLD = self._costFLD()
        self.costCYC = self._costCYC()
        self.costCSP = sum(
            [
                self.costLND,
                self.costTWR,
                self.costREC,
                self.costLFT,
                self.costHTF,
                self.costTES,
                self.costFLD, 
            ]
        )
        self.costTOT = sum(
            [
                self.costCSP, 
                self.costCYC
            ]
        )

        return self.costTOT
    def _costLND(self): # Land
        return 1e-6 * 2.5 * (self.A_field * 6 + 45000)
    def _costTWR(self): # Tower
        return 1e-6 * 157.44 * (self.H_tower ** 1.9174)
    def _costREC(self): # Receiver
        return 1e-6 * 37400 * self.A_receiver
    def _costLFT(self): # Lifts
        # return 1e-6 * 58.37 * self.H_lifts * self.m_dot_p
        return 31181446.6362307e-6
    def _costHTF(self): # Heat transfer fluid
        return 1e-6 * (1 + self.non_storage) * self.cp * self.m_htf
    def _costTES(self): # Thermal Energy Storage
        return 1e-6 * ((self.c_bin_h * self.A_bin_surf) + (self.c_bin_c * self.A_bin_surf) + self.c_losses)
    def _costFLD(self): # Solar Field
        return 1e-6 * (75 + 10) * self.A_field
    def _costCYC(self): # Power Block
        return 92.09


if __name__=='__main__': 

    efficiencys = np.linspace(0.3, 0.6, 20)

    df = pd.DataFrame()
    for eta in efficiencys: 
        model = costModel()
        costs = model.analysis(eta)
        df = pd.concat([df, pd.DataFrame([costs])], ignore_index=True)

    notInterested = ['eta_cyc', 'costCYC', 'costCSP', 'costTOT', 'totLCOE']
    colors = sns.color_palette("mako", len(df.columns)-len(notInterested))
    labels = {
        'costLND': 'Land', 
        'costTWR': 'Solar Tower', 
        'costREC': 'Receiver', 
        'costLFT': 'Particle Lifts', 
        'costHTF': 'Particles (bulk)', 
        'costTES': 'Particle Storage', 
        'costFLD': 'Solar Field', 
        'costCYC': 'Power Block', 
    }
    bottom = [0] * len(df)
    colorID = 0
    for key in df.columns: 
        if key in notInterested: continue

        plt.bar(df['eta_cyc']*100, df[key], color=colors[colorID], label=labels[key], bottom=bottom, zorder=5)
        bottom = [i + j for i, j in zip(bottom, df[key])]
        colorID += 1

    h, l = plt.gca().get_legend_handles_labels()
    h = h[::-1]
    l = l[::-1]

    plt.title('CSP Capital Cost by Power Block Efficiency')
    plt.xlabel(r'$\eta_{thermal}$ [%]')
    plt.ylabel('CSP Cost [M$]')
    plt.margins(x=0)
    plt.legend(h, l)
    plt.grid(zorder=1)
    plt.show()

    # print(list(df['eta_cyc']))
    # print(list(df['costCSP']))

    print(df)
    df.to_csv('test.csv')


