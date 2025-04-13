

/*
1) init variables
2) solve particle flow conditions
3) 
*/

// Calculate steady state temperature and heat loss profiles for a given mass flow and incident flux
void C_falling_particle_receiver::calculate_steady_state_soln(s_steady_state_soln& soln, double tol, bool init_from_existing, int max_iter)
{

    double tolTp = 0.1;  // Particle temperature tolerance [K] for convergence of solution iterations

    double P_amb = soln.p_amb;
    double T_amb = soln.T_amb;
    double v_wind_10 = soln.v_wind_10;
    double wdir = soln.wind_dir;
    double T_sky = soln.T_sky;
    double T_cold_in = soln.T_particle_cold_in;

    double v_wind = scale_wind_speed(v_wind_10);  // Wind speed at receiver height
    double dy = m_curtain_height / (m_n_y - 1);
    double dx = m_curtain_width / m_n_x;

    double Q_inc, Q_refl, Q_rad, Q_adv, Q_cond, Q_thermal, Q_imbalance;
    double Tp_out, T_particle_prop, T_cold_in_rec, cp, cp_cold, cp_hot, particle_density, err, hadv_with_wind, Twavg, Twmax, Twf;
    double qnetc_avg, qnetc_sol_avg, qnetw_avg, qnetw_sol_avg;
    double tauc_avg, rhoc_avg;

    Q_refl = Q_rad = Q_adv = Q_cond = Q_thermal = Q_imbalance = 0.0;
    Twavg = Twmax = Twf = Tp_out = hadv_with_wind = 0.0;
    qnetc_avg = qnetc_sol_avg = qnetw_avg = qnetw_sol_avg = 0.0;
    tauc_avg = rhoc_avg = 0.0;

    // Total solar power incident on curtain [W]
    Q_inc = sum_over_rows_and_cols(soln.q_dot_inc, true) * m_curtain_elem_area;  
    T_particle_prop = (m_T_htf_hot_des + T_cold_in) / 2.0; 
    particle_density = field_htfProps.dens(T_particle_prop, 1.0);
    
    // Particle specific heat  [J/kg-K] evaluated at average of current inlet temperature and design outlet temperature
    cp = field_htfProps.Cp(T_particle_prop) * 1000.0; 
    
    cp_cold = field_htfProps.Cp(T_cold_in) * 1000.0;
    cp_hot = field_htfProps.Cp(m_T_htf_hot_des) * 1000.0;
    
    // Inlet temperature to receiver accounting from loss from cold particle transport
    T_cold_in_rec = T_cold_in - m_deltaT_transport_cold; 

    double rhow_solar = 1.0 - m_cav_abs;
    double rhow_IR = 1.0 - m_cav_emis;

    bool rec_is_off = false;
    bool converged = false;
    bool soln_exists = (soln.T_particle_hot == soln.T_particle_hot) && (soln.converged);  // Does a solution already exist from a previous iteration?

    // Quasi-2D physics-based receiver model
    double vel_out, Tfilm, hadv, fwind, Rwall, tauc1, rhoc1;
    double qnet_ap, qnet_wf, hwconv, qadv, qtot, dh, Tcond_prev, Tcond_next;
    double Twfnew, Tpdiff, Twdiff, Twfdiff;
    util::matrix_t<double> Tp, Tw, Tpnew, Twnew;

    Rwall = 1.0 / m_cav_hext + m_cav_twall / m_cav_kwall;  // Cavity wall thermal resistance
    






    //--- Solve mass and momentum equations for curtain velocity, thickness, void fraction
    util::matrix_t<double> mdot_per_elem(m_n_x);  // Mass flow (kg/s) per element
    for (int i = 0; i < m_n_x; i++)
    {
        mdot_per_elem.at(i) = soln.m_dot_tot / m_n_x;  // TODO: Update to allow for multiple mass flow control zones
    }
    solve_particle_flow(mdot_per_elem, soln.phip, soln.vel, soln.thc);
    vel_out = calculate_mass_wtd_avg_exit(mdot_per_elem, soln.vel);  // Mass-weighted average exit velocity








    //--- Calculate curtain optical properties from solution for curtain void fraction and thickness
    soln.tauc.resize_fill(m_n_y, m_n_x, 0.0);
    soln.rhoc.resize_fill(m_n_y, m_n_x, 0.0);
    for (int j = 0; j < m_n_y; j++)
    {
        for (int i = 0; i < m_n_x; i++)
        {
            calculate_local_curtain_optical_properties(soln.thc.at(j, i), soln.phip.at(j, i), rhoc1, tauc1);
            soln.rhoc.at(j, i) = rhoc1;
            soln.tauc.at(j, i) = tauc1;
        }
    }
    tauc_avg = sum_over_rows_and_cols(soln.tauc, false) / (m_n_x * m_n_y);  // Average curtain transmittance
    rhoc_avg = sum_over_rows_and_cols(soln.rhoc, false) / (m_n_x * m_n_y);  // Average curtain reflectance

    //--- Initialize solutions for particle and wall temperatures
    Tp.resize_fill(m_n_y, m_n_x, T_cold_in_rec);        // Particle temperature [K]
    Tpnew.resize_fill(m_n_y, m_n_x, T_cold_in_rec);
    Tw.resize_fill(m_n_y, m_n_x, T_cold_in_rec);       // Back wall temperature [K]
    Twnew.resize_fill(m_n_y, m_n_x, T_cold_in_rec);

    //--- Set initial guess for particle and wall temperatures
    if (soln_exists && init_from_existing)
    {
        Tp = soln.T_p;
        Tw = soln.T_back_wall;
        Twf = soln.T_front_wall;
    }
    else
    {
        double qabs_approx, qnet_approx, dh_approx, rhoc_avg, flux_avg, Ec_avg, vf_to_ap;

        if (m_hadv_model_type == 0)
        {
            hadv = m_hadv_user;
            fwind = 1.0;
        }
        else if (m_hadv_model_type == 1) // Use Sandia's correlations for advective loss
        {
            Tfilm = 0.5 * (0.5 * (m_T_htf_hot_des + T_cold_in_rec) + T_amb);
            calculate_advection_coeff_sandia(vel_out, Tfilm, v_wind, wdir, P_amb, hadv, fwind);
            hadv = fmax(hadv, 0.0);
            hadv *= m_hadv_mult;
        }
        else  // TODO: Need an error here... eventually expand to include a lookup table
        {
            hadv = 0.0;
            fwind = 1.0;
        }
        hadv_with_wind = hadv * fwind;

        rhoc_avg = flux_avg = Ec_avg = 0.0;
        vf_to_ap = m_rad_model_type == 1 ? m_vf_curtain_ap_avg : m_vf_rad_type_0;
        for (int i = 0; i < m_n_x; i++)
        {
            Tp.at(0, i) = T_cold_in_rec;
            for (int j=0; j<m_n_y; j++)
            {
                if (j < m_n_y - 1)
                {
                    qabs_approx = (1.0 - soln.rhoc.at(j, i) - soln.tauc.at(j, i) + rhow_solar * soln.tauc.at(j, i) + rhow_solar * (1.0 - vf_to_ap) * soln.rhoc.at(j, i)) * soln.q_dot_inc.at(j, i); // Approximate solar energy absorbed by the particle curtain (W/m2)
                    qnet_approx = qabs_approx - hadv_with_wind * (Tp.at(j, i) - soln.T_amb) - m_curtain_emis * CSP::sigma * (pow(Tp.at(j, i), 4) - pow(soln.T_amb,4));  //Approximate net heat transfer rate using curtain temperature at prior element
                    dh_approx = qnet_approx * (dy / (soln.phip.at(j + 1, i) * soln.thc.at(j + 1, i) * soln.vel.at(j + 1, i) * particle_density));
                    Tp.at(j + 1, i) = fmax(T_cold_in_rec, Tp.at(j, i) + dh_approx / cp);
                    if (Tp.at(j + 1, i) < soln.T_amb)
                        Tp.at(j + 1, i) = soln.T_amb;
                }
                qnet_approx = (1.0 - rhow_solar) * (1.0 + rhow_solar * soln.rhoc.at(j, i)) * (soln.tauc.at(j, i) * soln.q_dot_inc.at(j, i));  // Approximate solar energy absorbed at back wall (W/m2)
                qnet_approx += (1.0 - rhow_IR)*(1.0 + rhow_IR*soln.rhoc.at(j,i)) * (m_curtain_emis * CSP::sigma * pow(Tp.at(j, i), 4));             // Add approximate IR radiative heat transfer incoming to the back wall (W/m2)
                Tw.at(j, i) = fmax(T_cold_in_rec, pow(qnet_approx / (m_cav_emis * CSP::sigma), 0.25));

                flux_avg += soln.q_dot_inc.at(j, i) / (m_n_x*m_n_y);
                rhoc_avg += soln.rhoc.at(j, i) / (m_n_x * m_n_y);
                Ec_avg += (m_curtain_emis * CSP::sigma * pow(Tp.at(j, i), 4)) / (m_n_x * m_n_y);
            }
        }
        qnet_approx = (1.0 - vf_to_ap) * m_cav_emis * (rhoc_avg * flux_avg + Ec_avg);
        Twf = fmax(T_cold_in_rec, pow(qnet_approx / (m_cav_emis * CSP::sigma), 0.25));  // Initial guess for front wall temperature


        //--- Calculate coefficient matrix for radiative exchange
        util::matrix_t<double> K_solar, Kinv_solar, K_IR, Kinv_IR;

        if (m_rad_model_type == 1)
        {
            calculate_coeff_matrix(soln.rhoc, soln.tauc, rhow_solar, K_solar, Kinv_solar);
            if (fabs(rhow_solar - rhow_IR) < 0.001)
            {
                K_IR = K_solar;
                Kinv_IR = Kinv_solar;
            }
            else
            {
                calculate_coeff_matrix(soln.rhoc, soln.tauc, rhow_IR, K_IR, Kinv_IR);
            }

        }

        //--- Initialize quantities needed in radiation models
        double qnet_ap_sol, qnet_wf_sol, Eap, Ewf;
        util::matrix_t<double> qnetc_sol, qnetw_sol, qnetc, qnetw, Ecf, Ecb, Ebw;
        qnetc_sol.resize_fill(m_n_y, m_n_x, 0.0);
        qnetw_sol.resize_fill(m_n_y, m_n_x, 0.0);
        qnetc.resize_fill(m_n_y, m_n_x, 0.0);
        qnetw.resize_fill(m_n_y, m_n_x, 0.0);
        Ecf.resize_fill(m_n_y, m_n_x, 0.0);
        Ecb.resize_fill(m_n_y, m_n_x, 0.0);
        Ebw.resize_fill(m_n_y, m_n_x, 0.0);

        

        //-- Solve radiative exchange equation for solar energy (this is independent of temperature)
        if (m_rad_model_type == 0)
        {
            Q_refl = 0.0;
            double jcback_sol, jw_sol, jcfront_sol;
            for (int i = 0; i < m_n_x; i++)
            {
                for (int j = 0; j < m_n_y; j++)
                {
                    jcback_sol = soln.tauc.at(j, i) * soln.q_dot_inc.at(j, i) / (1 - soln.rhoc.at(j, i) * rhow_solar);
                    jw_sol = rhow_solar * jcback_sol;
                    jcfront_sol = m_vf_rad_type_0 * (soln.rhoc.at(j, i) * soln.q_dot_inc.at(j, i) + soln.tauc.at(j, i) * jw_sol);
                    qnetc_sol.at(j, i) = (soln.q_dot_inc.at(j, i) - jcfront_sol) + (jw_sol - jcback_sol);
                    qnetw_sol.at(j, i) = (1.0-rhow_solar) * soln.tauc.at(j, i) * soln.q_dot_inc.at(j, i) / (1.0 - soln.rhoc.at(j, i) * rhow_solar);
                    if (j < m_n_y - 1)
                        Q_refl += jcfront_sol * m_curtain_elem_area;
                }
            }
        }
        else if (m_rad_model_type == 1)
        {
            //-- Solve radiative exchange equation for solar energy (this is independent of temperature)
            //   Here the energy "source" is solar flux reflected by the curtain on the first pass, and solar flux transmitted through the curtain and reflected by the back wall on the first pass
            //   This assumes that solar flux transmitted through the curtain hits the back wall at the same discretized element
            Ecf.fill(0.0);
            Ecb.fill(0.0);
            Ebw.fill(0.0);
            for (int i = 0; i < m_n_x; i++)
            {
                for (int j = 0; j < m_n_y; j++)
                {
                    Ecf.at(j,i) = soln.rhoc.at(j, i) * soln.q_dot_inc.at(j, i);         // Energy "source" at front curtain surface = reflected solar energy
                    Ebw.at(j,i) = rhow_solar * soln.tauc.at(j, i) * soln.q_dot_inc.at(j, i);  // Energy "source" at back wall surface = transmitted and reflected solar energy
                }
            }
            calculate_radiative_exchange(Ecf, Ecb, Ebw, 0.0, 0.0, K_solar, Kinv_solar, soln.rhoc, soln.tauc, rhow_solar, qnetc_sol, qnetw_sol, qnet_wf_sol, qnet_ap_sol);  // Calculates net incoming radiative energy to each element (total incoming - total outgoing)

            // Radiative exchange model provides the net incoming energy to each surface (net outgoing is -qnet). Now calculate the total net incoming solar energy
            for (int i = 0; i < m_n_x; i++)
            {
                for (int j = 0; j < m_n_y; j++)
                {
                    qnetc_sol.at(j,i) += (1 - soln.tauc.at(j, i)) * soln.q_dot_inc.at(j, i);
                    qnetw_sol.at(j,i) += soln.tauc.at(j, i) * soln.q_dot_inc.at(j, i);
                }
            }
            Q_refl = qnet_ap_sol * m_ap_area;   // Solar reflection loss [W]
        }
        qnetc_sol_avg = sum_over_rows_and_cols(qnetc_sol, true) / (m_n_x * (m_n_y - 1));  // Average net incoming solar energy to the particle curtain
        qnetw_sol_avg = sum_over_rows_and_cols(qnetw_sol, true) / (m_n_x * (m_n_y - 1));  // Average net incoming solar energy to the back wall

        //--- Temperature solution iterations
        for (int q = 0; q < max_iter; q++)
        {
            Q_rad = Q_adv = Q_cond = Q_thermal = 0.0;
            Tpdiff = Twdiff = Twfdiff = 0.0;

            Twavg = sum_over_rows_and_cols(Tw, true) / (m_n_x * (m_n_y-1));  // Current average back wall temperature (neglecting last element)
            Twmax = max_over_rows_and_cols(Tw, true);  // Current maximum back wall temperature (neglecting last element)

            //-- Calculate advection loss coefficient
            if (m_hadv_model_type == 0)
            {
                hadv = m_hadv_user;
                fwind = 1.0;
            }
            else if (m_hadv_model_type == 1) // Use Sandia's correlations for advective loss
            {
                Tp_out = calculate_mass_wtd_avg_exit(mdot_per_elem, Tp);
                Tfilm = 0.5 * (0.5 * (Tp_out + T_cold_in_rec) + T_amb);
                calculate_advection_coeff_sandia(vel_out,  Tfilm, v_wind, wdir, P_amb, hadv, fwind);
                hadv = fmax(hadv, 0.0);
                hadv *= m_hadv_mult;
            }
            else  // TODO: Need an error here... eventually expand to include a lookup table
            {
                hadv = 0.0;
                fwind = 1.0;
            }
            hadv_with_wind = hadv * fwind;


            //-- Solve IR radiative exchange and calculate net incoming radiative energy to each curtain and wall element
            if (m_rad_model_type == 0)
            {
                double Ew, Ec, jcback_IR, jw_IR, jcfront_IR;
                for (int i = 0; i < m_n_x; i++)
                {
                    for (int j = 0; j < m_n_y; j++)
                    {
                        Ew = m_cav_emis * CSP::sigma * pow(Tw.at(j, i), 4);
                        Ec = m_curtain_emis * CSP::sigma * pow(Tp.at(j, i), 4);
                        jcback_IR = (Ec + soln.rhoc.at(j, i) * Ew) / (1 - soln.rhoc.at(j, i) * rhow_IR);
                        jw_IR = Ew + rhow_IR * jcback_IR;
                        jcfront_IR = m_vf_rad_type_0 * (Ec + soln.tauc.at(j, i) * jw_IR);
                        qnetc.at(j, i) = qnetc_sol.at(j, i) - jcfront_IR + (jw_IR - jcback_IR);
                        qnetw.at(j, i) = qnetw_sol.at(j, i) + ((1 - rhow_IR) * Ec - (1 - soln.rhoc.at(j, i)) * Ew) / (1.0 - soln.rhoc.at(j, i) * rhow_IR);
                        if (j < m_n_y - 1)
                            Q_rad += jcfront_IR * m_curtain_elem_area;
                    }
                }
            }
            else if (m_rad_model_type == 1)
            {
                Eap = CSP::sigma * pow(T_sky, 4);
                Ewf = m_cav_emis * CSP::sigma * pow(Twf, 4);  // Front cavity wall
                Ecf.fill(0.0);
                Ecb.fill(0.0);
                Ebw.fill(0.0);
                for (int i = 0; i < m_n_x; i++)
                {
                    for (int j = 0; j < m_n_y; j++)
                    {
                        Ecf.at(j,i) = m_curtain_emis * CSP::sigma * pow(Tp.at(j, i), 4); // Front curtain surface
                        Ecb.at(j, i) = Ecf.at(j, i);                                     // Back curtain surface
                        Ebw.at(j,i) = m_cav_emis * CSP::sigma * pow(Tw.at(j, i), 4);     // Back wall surface
                    }
                }
                calculate_radiative_exchange(Ecf, Ecb, Ebw, Eap, Ewf, K_IR, Kinv_IR, soln.rhoc, soln.tauc, rhow_IR, qnetc, qnetw, qnet_wf, qnet_ap);    // Calculates net incoming radiative energy to each element (total incoming - total outgoing)
                Q_rad = qnet_ap * m_ap_area;    // IR radiation loss [W]

                //--- Combine solar and IR exchange
                qnet_ap += qnet_ap_sol;
                qnet_wf += qnet_wf_sol;
                qnetc = matrix_addition(qnetc, qnetc_sol);
                qnetw = matrix_addition(qnetw, qnetw_sol);

            }
            qnetc_avg = sum_over_rows_and_cols(qnetc, true) / (m_n_x * (m_n_y - 1));
            qnetw_avg = sum_over_rows_and_cols(qnetw, true) / (m_n_x * (m_n_y - 1));

            //--- Calculate new back wall and particle temperature solutions, Q_adv, Q_thermal, and Q_cond
            for (int i = 0; i < m_n_x; i++)
            {
                for (int j = 0; j < m_n_y; j++)
                {
                    hwconv = m_include_back_wall_convection ? calculate_wall_convection_coeff(soln.vel.at(j, i), (j + 0.5) * dy, 0.5 * (Tw.at(j, i) + Tp.at(j, i)), P_amb) : 0.0;

                    // Solve for particle temperature at next vertical node
                    qadv = hadv_with_wind* (Tp.at(j, i) - T_amb);
                    if (j < m_n_y - 1)
                    {
                        qtot = qnetc.at(j, i) - qadv - hwconv * (Tp.at(j, i) - Tw.at(j, i));  // Net heat transfer rate into particles [W/m2]
                        dh = qtot * (dy / (soln.phip.at(j + 1, i) * soln.thc.at(j + 1, i) * soln.vel.at(j + 1, i) * particle_density));
                        Tpnew.at(j + 1, i) = Tpnew.at(j, i) + dh / cp;
                        Q_adv += qadv * m_curtain_elem_area;            // Advection loss [W] from this curtain element
                        Q_thermal += dh * mdot_per_elem.at(i);          // Energy into particles [W] in this curtain elemtn
                    }
                    Tpdiff = fmax(Tpdiff, fabs(Tpnew.at(j, i) - Tp.at(j, i)));

                    // Solve for back wall temperature at current node
                    Tcond_prev = Tcond_next = 0.0;
                    if (m_include_wall_axial_conduction)
                    {
                        if (j == m_n_y - 1)
                            Tcond_next = Twnew.at(j - 1, i);  // Tw(j+1) = Tw(j-1) for heat flux = 0 boundary condition at last node
                        else if (i == 0)  // First iteration doesn't have a good value for Tw(j+1) already defined
                            Tcond_next = (j == 0) ? Tw.at(j, i) : Tw.at(j, i) + (Twnew.at(j, i) - Twnew.at(j - 1, i));  //As a first guess, assume wall temperature rise from (j to j+1) is the same as that from (j-1 to j)
                        else
                            Tcond_next = Tw.at(j + 1, i);
                        Tcond_prev = (j > 0) ? Twnew.at(j - 1, i) : Tcond_next;
                    }
                    Twnew.at(j, i) = calculate_passive_surface_T(Tw.at(j, i), qnetw.at(j, i), hwconv, Tp.at(j, i), T_amb, m_include_wall_axial_conduction, Tcond_prev, Tcond_next);
                    if (j < m_n_y - 1)
                    {
                        Q_cond += ((Tw.at(j, i) - T_amb) / Rwall) * m_back_wall_elem_area;   // Conduction loss through back wall element [W]
                        Twdiff = fmax(Twdiff, fabs(Twnew.at(j, i) - Tw.at(j, i)));
                    }
                    
                }
            }

            //--- Calculate new front wall temperature solution and conduction loss
            Twfdiff = Twfnew = 0.0;
            if (m_rad_model_type > 0)
            {
                Twfnew = calculate_passive_surface_T(Twf, qnet_wf, 0.0, 0.0, T_amb, false, 0.0, 0.0);
                Q_cond += ((Twf - T_amb) / Rwall) * m_cav_front_area;
                Twfdiff = fabs(Twfnew - Twf);
            }

            //--- Check convergence and update temperature solution
            Q_imbalance = Q_inc - Q_refl - Q_rad - Q_adv - Q_cond - Q_thermal;
            err = Q_imbalance / Q_inc;  // 
            if (Tpdiff < tolTp && fabs(err) < tol)
            {
                converged = true;
                soln.T_p = Tpnew;
                soln.T_back_wall = Twnew;
                soln.T_front_wall = Twfnew;
                soln.T_back_wall_avg = Twavg;
                soln.T_back_wall_max = Twmax;
                Tp = Tpnew;
                Tw = Twnew;
                Twf = Twfnew;
                break;
            }

            Tp = Tpnew;
            Tw = Twnew;
            Twf = Twfnew;

            // Stop iterations for very low outlet temperature, or if the solution in the previous iteration failed 
            if (Q_thermal != Q_thermal)
                break;

            Tp_out = calculate_mass_wtd_avg_exit(mdot_per_elem, Tp);  // Mass-weighted average particle exit temperature from receiver [K]
            if (Tp_out < 0.0)
                break;
        }

        if (!converged)
            rec_is_off = true;  // Shut off receiver if temperature solution failed

        Tp_out = calculate_mass_wtd_avg_exit(mdot_per_elem, Tp);  // Mass-weighted average particle exit temperature from receiver [K]
    }

    double Tp_out_after_transport = Tp_out - m_deltaT_transport_hot;  // Outlet temperature accounting from loss from hot particle transport

    double Q_dot_transport_loss_hot = soln.m_dot_tot * cp * m_deltaT_transport_hot;
    double Q_dot_transport_loss_cold = soln.m_dot_tot * cp * m_deltaT_transport_cold;

    if (Tp_out <= T_cold_in || Q_thermal != Q_thermal)
    {
        rec_is_off = true;
    }


	// Save solution
    soln.Q_inc = Q_inc;
    soln.Q_thermal = Q_thermal - Q_dot_transport_loss_cold - Q_dot_transport_loss_hot;
    soln.Q_thermal_without_transport = Q_thermal;
	soln.Q_refl = Q_refl;
	soln.Q_rad = Q_rad;
	soln.Q_adv = Q_adv;
    soln.Q_cond = Q_cond;
    soln.Q_transport = Q_dot_transport_loss_cold + Q_dot_transport_loss_hot;
    soln.eta = soln.Q_inc>0 ? soln.Q_thermal_without_transport / soln.Q_inc : 0.0;
    soln.eta_with_transport = soln.Q_inc > 0 ? soln.Q_thermal / soln.Q_inc : 0.0;
    soln.hadv = hadv_with_wind;
    soln.converged = converged;
	soln.rec_is_off = rec_is_off;

    soln.T_particle_hot = Tp_out_after_transport;
    soln.T_particle_hot_rec = Tp_out;

    soln.tauc_avg = tauc_avg;
    soln.rhoc_avg = rhoc_avg;
    soln.qnetc_sol_avg = qnetc_sol_avg;
    soln.qnetw_sol_avg = qnetw_sol_avg;
    soln.qnetc_avg = qnetc_avg;
    soln.qnetw_avg = qnetw_avg;

	return;

}







