
// Calculate mass flow rate needed to achieve target outlet temperature (m_T_salt_hot_target) given incident flux profiles
void C_falling_particle_receiver::solve_for_mass_flow(s_steady_state_soln &soln)
{

	bool soln_exists = (soln.m_dot_tot == soln.m_dot_tot);
    double T_particle_prop = (m_T_particle_hot_target + soln.T_particle_cold_in) / 2.0; // Temperature for particle property evaluation [K].  
    double cp = field_htfProps.Cp(T_particle_prop) * 1000.0;   //[J/kg-K]

    double T_cold_in_rec = soln.T_particle_cold_in - m_deltaT_transport_cold;    // Temperature at the receiver inlet (after cold particle transport)
    double T_target_out_rec = m_T_particle_hot_target + m_deltaT_transport_hot;  // Temperature at the receiver exit (before hot particle transport)

    double err = -999.9;				//[-] Relative outlet temperature error
    double tol = 1.0e-4;
    int nmax = 50;

    bool allow_Tmax_stopping = true;   // Allow mass flow iteration loop to stop based on estimated upper bound of particle outlet temperature (assumes efficiency increases monotonically with mass flow)
    if (soln.require_operating_mode && soln.required_mode == C_csp_collector_receiver::E_csp_cr_modes::ON && m_fixed_mode_mflow_method == 1)
        allow_Tmax_stopping = false;      // If receiver can't reach target exit temperature the solution will return the mass flow required for the maximum exit temperature


    //--- Set initial guess for particle flow
    double m_dot_guess, m_dot_guess_new;
	if (soln_exists)  // Use existing solution as initial guess
	{
		m_dot_guess = soln.m_dot_tot;
	}
	else  // Set initial guess for mass flow solution
	{
        double Qinc = sum_over_rows_and_cols(soln.q_dot_inc, true) * m_curtain_elem_area;  // Total solar power incident on curtain [W]
        double eta_guess = m_model_type == 0 ? m_fixed_efficiency : 0.85;
        m_dot_guess = eta_guess * Qinc / (cp * (T_target_out_rec - T_cold_in_rec));	//[kg/s] Particle mass flow rate
	}



    //--- Solve for mass flow
    //    Note: Outlet temperature is non-monotonic with respect to mass flow because the curtain transparency changes with mass flow 
    //    The solution here relies on mass flow bounds that are derived assuming that there is only one maximum in the behavior of particle outlet temperature vs. mass flow rate
    //    The efficiency (Q_thermal / Q_inc) is assumed to increase monotonically to with respect to mass flow and the upper bound on efficiency is used to calculate an upper bound on outlet temperature
    //    Two mass flow solutions will exist that can reach the target outlet tempeature (for those cases where the target outlet temperature can be reached at all). The solution with higher mass flow is returned here. 

    double lower_bound = 0.0;                   // Lower bound for particle mass flow rate (kg/s)
    double upper_bound = 1e10;                  // Upper bound for particle mass flow rate  (kg/s)
    double upper_bound_eta = 1.0;               // Upper bound for receiver efficiency (assumed to occur at upper bound for mass flow)
    bool is_upper_bound = false;                // Has upper bound been defined?
    bool is_lower_bound_above_Ttarget = false;  // Has a solution already been found with outlet T > target T?

    double bound_tol = 0.01;                  // Stopping tolerance based on mass flow bounds (typically solution will terminate because of calculated maximum outlet T first)
    double Tout_tol = 1;
    double lower_bound_Tout = std::numeric_limits<double>::quiet_NaN();  // Outlet temperature at current lower bound for mass flow
    double upper_bound_Tout = std::numeric_limits<double>::quiet_NaN();  // Outlet temperature at current upper bound for mass flow

    double Tout_max = std::numeric_limits<double>::quiet_NaN();  // Current maximum possible outlet temperature [K] (based on upper bound for efficiency, lower bound for mass flow)
    double eta_max_sim = std::numeric_limits<double>::quiet_NaN();  // Maximum simulated receiver efficiency

    util::matrix_t<double> mflow_history, Tout_history, eta_history;
    util::matrix_t<bool> converged_history;
    mflow_history.resize_fill(nmax, 0.0);       // Mass flow iteration history
    Tout_history.resize_fill(nmax, 0.0);        // Outlet temperature iteration history
    eta_history.resize_fill(nmax, 0.0);         // Receiver efficiency iteration history
    converged_history.resize_fill(nmax, false); // Steady state solution convergence history
   

    int qq = -1;
    bool converged = false;
    bool init_from_existing = false;  // Use current solution for particle/wall temperatures as the initial guess for the next steady state solution
	while (!converged)
	{
		qq++;

        //-- Solve model at current mass flow guess
		soln.m_dot_tot = m_dot_guess;
		calculate_steady_state_soln(soln, tol, init_from_existing, 50);   // Solve steady state thermal model
		err = (soln.T_particle_hot - m_T_particle_hot_target) / m_T_particle_hot_target;
        mflow_history.at(qq) = soln.m_dot_tot;
        Tout_history.at(qq) = soln.T_particle_hot_rec;   // Outlet temperature from receiver (before hot particle transport)
        eta_history.at(qq) = soln.eta;                   // Efficiency not including transport losses
        converged_history.at(qq) = soln.converged;
        eta_max_sim = (qq == 0) ? soln.eta : fmax(eta_max_sim, soln.eta);

        init_from_existing = false;
        if (qq>0 && !soln.rec_is_off && std::abs(soln.T_particle_hot_rec - Tout_history.at(qq - 1)) < 20)  
            init_from_existing = true;
      

        //--- Check mass flow convergence
        if (std::abs(err) < tol)  // Solution outlet temperature is at the target
        {
            if (is_lower_bound_above_Ttarget)  // A different solution has already been found with a lower mass flow and an outlet T above the target
            {
                converged = true;
                break;
            }
            else  // Current mass flow produces the target outlet T, but can't be sure that it's the higher of the two possible mass flow solutions
            {
                s_steady_state_soln soln_candidate = soln;  // Store the current solution as a possible candidate solution
                soln.m_dot_tot = (1 - 0.015) * soln.m_dot_tot; // If this is the higher mass flow solution than a decrease in mass flow should increase outlet temperature
                calculate_steady_state_soln(soln, tol, true, 50);
                bool is_correct_soln = (soln.T_particle_hot_rec > soln_candidate.T_particle_hot_rec);
                soln = soln_candidate;
                if (is_correct_soln)
                {
                    converged = true;
                    break;
                }
                else  // Decreasing mass flow decreased the outlet temperature, this is the lower of the two mass flow solutions -> Set lower bound and continue iterating
                {
                    lower_bound = fmax(lower_bound, soln.m_dot_tot);
                    is_lower_bound_above_Ttarget = true;
                    m_dot_guess = is_upper_bound ? 0.5 * (lower_bound + upper_bound) : 1.05 * m_dot_guess;
                    continue;
                }
            }
        }


        //--- Update mass flow bounds

        // If efficiency is negative, this is always a lower bound for mass flow
        if (soln.eta < 0.0)
            lower_bound = soln.m_dot_tot;


        // If outlet temperature is above the target this is always a lower bound for mass flow 
        if (soln.T_particle_hot_rec > T_target_out_rec)
        {
            lower_bound = soln.m_dot_tot;
            is_lower_bound_above_Ttarget = true;  
        }

        // If outlet temperature is below the target, compare current solution with previously stored solutions to identify bounds
        else if (soln.converged)
        {
            for (int i = 0; i < qq; i++)  
            {
                if (converged_history.at(i))
                {
                    // Any point with an exit temperature below the target is a lower bound if another point has been sampled with higher mass flow and a higher outlet T
                    if (soln.m_dot_tot < mflow_history.at(i) && soln.T_particle_hot_rec < Tout_history.at(i))
                    {
                        lower_bound = soln.m_dot_tot;
                        lower_bound_Tout = soln.T_particle_hot_rec;
                    }
                    if (mflow_history.at(i) > lower_bound && Tout_history.at(i) < T_target_out_rec && mflow_history.at(i) < soln.m_dot_tot && Tout_history.at(i) < soln.T_particle_hot_rec)
                    {
                        lower_bound = mflow_history.at(i);
                        lower_bound_Tout = Tout_history.at(i);
                    }

                    // Any point with an exit temperature below the target is an upper bound if another point has been sampled with lower mass flow and higher outlet temperature 
                    if (soln.m_dot_tot > mflow_history.at(i) && soln.T_particle_hot_rec < Tout_history.at(i))
                    {
                        upper_bound = soln.m_dot_tot;
                        upper_bound_eta = soln.eta;
                        upper_bound_Tout = soln.T_particle_hot_rec;
                        is_upper_bound = true;
                    }
                    if (mflow_history.at(i) < upper_bound && Tout_history.at(i) < T_target_out_rec && mflow_history.at(i) > soln.m_dot_tot && Tout_history.at(i) < soln.T_particle_hot_rec)
                    {
                        upper_bound = mflow_history.at(i);
                        upper_bound_eta = eta_history.at(i);
                        upper_bound_Tout = Tout_history.at(i);
                        is_upper_bound = true;
                    }    
                }

            }
        }


        //--- Next solution guess
        Tout_max = 5000;
        if (lower_bound > 0.001)
            Tout_max = T_cold_in_rec + (upper_bound_eta * soln.Q_inc) / (cp * lower_bound);  //  Maximum possible outlet temperature at current lower bound for flow and upper bound for efficiency

        if (soln.eta < 0.0)  // If efficiency at current mass flow guess was negative, try again with a substantially higher flow
        {
            m_dot_guess_new = 2 * soln.m_dot_tot;
        }
        else if (Tout_max < T_target_out_rec - 0.01)  // If maximum possible outlet T is below the target, then the iterations need to find the maximum outlet T instead of the target outlet T
        {
            m_dot_guess_new = is_upper_bound ? 0.5 * (lower_bound + upper_bound) : 1.5 * soln.m_dot_tot;
        }
        else  // If it's still possible that the receiver can achieve the target exit temperature, guess new mass flow based on the value needed to hit the target
        {
            m_dot_guess_new = soln.Q_thermal_without_transport / (cp * (T_target_out_rec - T_cold_in_rec));   //[kg/s]
            // Solution can converge slowly, after a few iterations switch to approximation for mass flow using linear approximation for efficiency from last two guesses
            if (qq >= 2 && eta_history.at(qq) > 0 && eta_history.at(qq - 1) > 0 && fabs(eta_history.at(qq) - eta_history.at(qq - 1)) < 0.3)
            {
                double c1 = (eta_history.at(qq) - eta_history.at(qq - 1)) / (mflow_history.at(qq) - mflow_history.at(qq - 1));  // Slope of linear equation for eta = f(m)
                double c2 = soln.eta - c1 * soln.m_dot_tot;                                                                     // Intercept of linear equation for eta = f(m)
                m_dot_guess_new = (soln.Q_inc * c2) / (cp * (T_target_out_rec - T_cold_in_rec) - c1 * soln.Q_inc);              // Solution for m from: (c1*m+c2)*Qinc = m*Cp*dT
            }

            // Limit downward changed in mass flow (avoids excessive under-shoot that can happen with very low power)
            if (m_dot_guess_new < soln.m_dot_tot)
                m_dot_guess_new = fmax(0.5 * soln.m_dot_tot, m_dot_guess_new);
        }

        //-- Check next solution guess relative to bounds
        if (is_upper_bound && (m_dot_guess_new <= lower_bound || m_dot_guess_new >= upper_bound))  // New guess is out of bounds and lower/upper bounds are both defined
            m_dot_guess_new = 0.5 * (lower_bound + upper_bound);
        else if (m_dot_guess_new <= lower_bound)  // New guess is below lower bound with no defined upper bound
            m_dot_guess_new = (m_dot_guess_new < m_dot_guess) ? fmax(1.5 * lower_bound, 0.5 * m_dot_guess) : 1.5 * m_dot_guess;

        if (fabs((m_dot_guess_new - soln.m_dot_tot) / soln.m_dot_tot) < 0.001)  // Next guess is identical to the last one (this can occur with bisection when the new solution at the midpoint doesn't change the known bounds)
            m_dot_guess_new *= 1.1;


        m_dot_guess = m_dot_guess_new;


        //--- Stopping criteria
        if (m_dot_guess < 1.E-5 || qq >= nmax-1)
        {
            soln.rec_is_off = true;
            break;
        }

        if (soln.Q_thermal != soln.Q_thermal) 
        {
            soln.rec_is_off = true;
            break;
        }

        // Stop solution if outlet temperature was not achieved and mass flow bounds are within tolerance
        if (is_upper_bound && (upper_bound - lower_bound) <= bound_tol * lower_bound && fabs(lower_bound_Tout - upper_bound_Tout)<=Tout_tol)
        {
            // Shut off receiver unless the operating mode is required to remain ON. 
            if (!soln.require_operating_mode || soln.required_mode != C_csp_collector_receiver::E_csp_cr_modes::ON || m_fixed_mode_mflow_method != 1)
                soln.rec_is_off = true; 
            break;
        }

        // Stop solution if the maximum possible particle outlet temperature is below the target
        // This assumes that the efficiency (Q_thermal / Q_inc) increases monotonically with mass flow for fixed solar incidence and ambient conditions
        if (allow_Tmax_stopping && lower_bound > 0.001)
        {
            Tout_max = T_cold_in_rec + (upper_bound_eta * soln.Q_inc) / (cp * lower_bound);
            if (Tout_max < T_target_out_rec -0.01)
            {
                soln.rec_is_off = true;
                break;
            }

        }

	}

    // Solve at fixed mass flow if receiver cannot achieve exit temperature, but is required to remain ON 
    if (soln.rec_is_off && soln.require_operating_mode && soln.required_mode == C_csp_collector_receiver::E_csp_cr_modes::ON && m_fixed_mode_mflow_method == 0)
    {
        soln.m_dot_tot = m_m_dot_htf_fixed;
        calculate_steady_state_soln(soln, tol, false, 50);
        soln.rec_is_off = false;
    }

    // Require final solution to have positive energy to particles, regardless of operating state requirements
    if (soln.Q_thermal < 0.0)
        soln.rec_is_off = true; 

	return;
}

