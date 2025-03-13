
#include <cmath>

#ifndef _CSPGEN3
#define _CSPGEN3

class cycleObj {
public: 

    double cycle_eta;                   // [-]
    double W_dot_net;                   // [MWe]
    double m_dot_phx;                   // [MWe]
    double W_dot_cool;                  // [MWe]
    double T_phx_i;                     // [K]
    double T_phx_o;                     // [K]
    double HTR_capital_cost;            // [$]
    double LTR_capital_cost;            // [$]
    double PHX_capital_cost;            // [$]
    double air_cooler_capital_cost;     // [$]
    double compressor_capital_cost;     // [$]
    double recompressor_capital_cost;   // [$]
    double turbine_capital_cost;        // [$]

}; 

class cspGen3CostModel {
public: 

    cspGen3CostModel();                             // Class initializer. Memory is shared and modified between optimizations. 
    ~cspGen3CostModel() = default;                  // Default destructor. Memory is out-of-scope after the optimization is completed. 
    void designRoutine(cycleObj* cycle, double SM); // Accepts a power block and solar multiple to design a CSP plant and calculate LCOE. 

    // decision variables
    double solar_multiple;  // [-]      ratio of solar field installed power to power block design point

    // derived parameters    
    double W_dot_thm;       // [MWt]    power cycle thermal input
    double W_dot_rec;       // [MWt]    power required at receiver
    double W_dot_less;      // [MWe]    net power, less parasitics
    double W_elec_annual;   // [kW-h]   annual electricity produced

    struct cycle { // power block data structure

        // power block parameters
        double T_phx_i;         // [K]   pHX particle inlet temperature
        double T_phx_o;         // [K]   pHX particle outlet temperature
        double efficiency;      // [-]   cycle efficiency
        double W_dot_net;       // [MWe] cycle design power output
        double phx_height;      // [m]   primary heat exchanger height

        cycle() {
            T_phx_i = T_phx_o = efficiency = W_dot_net = phx_height = 0; 
        }; 
    } s_cycle; 

    struct costs { // capital costs data structure

        // CSP capital costs
        double solar_tower;                 // [$] CSP Gen3 Solar Tower capital cost
        double solar_field;                 // [$] CSP Gen3 Solar Field capital cost
        double falling_particle_receiver;   // [$] CSP Gen3 Falling Particle Receiver capital cost
        double particles;                   // [$] bulk cost of particles
        double particle_losses;             // [$] incurred cost due to particle loss / attrition
        double particle_storage;            // [$] particle storage bins and insulation capital cost
        double particle_lifts;              // [$] particle transportation capital cost
        double land;                        // [$] bulk cost of land required for power plant

        // Cycle capital costs
        double HTR_capital_cost;            // [$] high temperature recuperator capital cost
        double LTR_capital_cost;            // [$] low temperature recuperator capital cost
        double PHX_capital_cost;            // [$] primary heat exchanger capital cost
        double air_cooler_capital_cost;     // [$] air cooler capital cost
        double compressor_capital_cost;     // [$] primary compressor capital cost
        double recompressor_capital_cost;   // [$] recompressor capital cost
        double turbine_capital_cost;        // [$] turbine capital cost

        // Total capital, maintenance, and LCOE
        double total_capital;               // [$]       total expected capital cost of plant
        double annual_maintenance;          // [$/year]  expected O&M annual costs
        double levelized_cost_of_energy;    // [$/MWe-h] CSP Gen3 levelized cost of energy
    
        costs() {
            solar_tower = solar_field = falling_particle_receiver = land = 
                particles = particle_storage = particle_lifts = particle_losses = 0; 

            HTR_capital_cost = LTR_capital_cost = PHX_capital_cost = air_cooler_capital_cost = 
                compressor_capital_cost = recompressor_capital_cost = turbine_capital_cost = 0; 

            annual_maintenance = total_capital = levelized_cost_of_energy = 0; 
        }; 
    } s_costs; 

    struct financing { // financing data structure

        // assumed lifetime, maintenance, and financing rates
        double discount_rate = 0.070;       // [-] cost of financing         (Albrecht, 2019)
        double construction  = 0.060;       // [-] construction costs        (Albrecht, 2019)
        double contingency   = 0.100;       // [-] unexpected costs          (Albrecht, 2019)
        double indirect      = 0.130;       // [-] indirect cost of capital  (Albrecht, 2019)
        double inflation     = 0.025;       // [-] average rate of inflation
        double lifetime      = 30.00;       // [years] total design lifetime
        double maintenance   = 40.00;       // [$/kWe-year] O&M costs        (Albrecht, 2019)

        // financing derived parameters
        double real_discount_rate;          // [-] calculated using discount rate and inflation
        double capital_recovery_factor;     // [-] calculated using real discount rate and assumed lifetime

        financing() {
            real_discount_rate = (1 + discount_rate) / (1 + inflation); // [-]
            capital_recovery_factor = 
                real_discount_rate * pow(1 + real_discount_rate, lifetime)
                / (pow(1 + real_discount_rate, lifetime) - 1); 
        }; 
    } s_financing; 

    struct parasitics { // net power parasitics data structure

        // net power parasitics
        double field;   // [MWe] parasitics from heliostat tracking
        double lifts;   // [MWe] parasitics from particle lifts
        double cooler;  // [MWe] parasitics from power block air cooler

        parasitics() {
            field = lifts = cooler = 0; 
        }; 
    } s_parasitics; 

    struct storage { // thermal energy storage data structure

        // thermal energy storage capacity
        double hours_of_capacity;   // [hours] 
        double capacity_factor;     // [-]

        struct warm { // warm storage

            // warm particle storage dimensions and performance
            double height;      // [m]
            double radius;      // [m]
            double volume;      // [m2]
            double temperature; // [K]
            double efficiency;  // [-]

            warm() {
                height = radius = volume = temperature = 0; 
            }
        } s_warm; 

        struct cold { // cold storage

            // cold particle storage dimensions and performance
            double height;      // [m]
            double radius;      // [m]
            double volume;      // [m2]
            double temperature; // [K]
            double efficiency;  // [-]

            cold() {
                height = radius = volume = temperature = 0; 
            }
        } s_cold;

        storage() {
            hours_of_capacity = capacity_factor = 0; 
        }; 
    } s_storage; 

    struct particles { // thermal carrier data structure

        // parameters derived from particle
        double cost_per_kg;     // [$/kg]   particle bulk cost
        double angle_of_repose; // [rad]    particle repose angle
        double bulk_density;    // [kg/m3]  particle bulk density

        // parameters derived from cycle
        double m_particles;     // [kg]     bulk particle mass
        double m_dot_phx;       // [kg/s]   mass flow through phx
        double m_dot_rec;       // [kg/s]   mass flow through receiver

        // assumed non-storage particle factor
        double non_storage = 0.05; // [-] 

        particles() {
            cost_per_kg = angle_of_repose = bulk_density
                 = m_particles = m_dot_phx = m_dot_rec = 0; 
        }; 
    } s_particles; 

    struct field { // solar field data structure

        // ratio of receiver thermal input to power cycle thermal input
        double solar_multiple;  // [-] solar multiple determines hours of energy storage

        // assumed solar field parameters
        double heliostat_cost_per_area = 75; // [$/m2]  cost per unit area of heliostat           (US DOE, 2012)
        double prep_cost_per_area = 10.0;    // [$/m2]  cost per unit area of land preparation    (US DOE, 2012)
        double land_cost_per_area = 2.47;    // [$/m2]  cost per unit area of land                (Mehos, 2016)
        double tracking_power = 0.0055;      // [kW/kW] heliostat tracking power gross fraction   (NREL SAM, 2024)

        // derived solar field parameters
        double area_heliostats; // [m2] total heliostat surface area required
        double area_total_land; // [m2] total land required

        field() {
            area_heliostats = area_total_land = solar_multiple = 0; 
        };
    } s_field; 

    struct tower { // solar tower data structure

        // solar tower dimensions
        double height; // [m] tower height
        double radius; // [m] tower radius

        tower() {
            height = radius = 0; 
        };
    } s_tower; 

    struct receiver { // falling particle receiver data structure

        // falling particle receiver dimensions and performance
        double height;               // [m]  flat plate receiver height 
        double width;                // [m]  flat plate receiver width
        double area_aperature;       // [m2] aperature area = receiver area
        double aspect_ratio;         // [-]  receiver height / receiver width
        double particle_loss_factor; // [-]  particle loss from open air receiver
        double efficiency;           // [-]  receiver efficiency

        receiver() {
            height = width = area_aperature = aspect_ratio = 0; 
            particle_loss_factor = efficiency = 0; 
        };
    } s_receiver; 

    struct lifts { // particle transportation data structure

        // particle transportation / lift parameters
        double height;      // [m] total combined lift height
        double efficiency;  // [-] lift electrical efficiency

        lifts() {
            height = efficiency = 0; 
        };
    } s_lifts; 

private: 

    void powerReceiver();           // Calculates power required at the receiver. 
    void sizeEquipment();           // Sizes CSP Gen3 equipment (solar tower, etc). 
    double costLand();              // Calculates cost of the total land required.
    double costTower();             // Calculates cost of the solar tower. 
    double costField();             // Calculates cost of the solar field / heliostats. 
    double costLifts();             // Calculates cost of the particle lifts. 
    double costStorage();           // Calculates cost of the thermal energy storage. 
    double costReceiver();          // Calculates cost of the falling particle receiver. 
    double costParticles();         // Calculates cost of bulk particles required. 
    double costParticleLosses();    // Calculates incurred cost of particle loss / attrition. 

}; 


#endif // _CSPGEN3

