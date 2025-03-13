
import threading

import multiprocessing as mp
import scipy.optimize as opt
import numpy as np
import queue
import csv
import sys
import os

from parameters import designParameters
from functools import partial 
from datetime import datetime
from copylot import CoPylot

def testCase(threads=8, save=False): 

    cp = CoPylot()
    dp = designParameters(cp)
    setStudyParameters(dp)
    setStudyVariables(dp, factorlosses=True)
    results = runInstance(cp, dp, threads)

    if save: saveStudyResults(results | dp.get())

    print("")
    print("Single SolarPILOT Study:")
    print(f"{'Power to HTF':.<20}{results['Power absorbed by HTF']/1e3:.>20.2f} [MWt]")
    print(f"{'Total Plant Cost':.<20}{results['Total plant cost']/1e6:.>20.2f} [$MM]")
    print("")
def saveStudyResults(results, csvfile=f'test.csv'): 
    # Saving all design parameters and study summary
    ordered = sorted(results.keys())
    csvfile = os.path.join(os.getcwd(), csvfile)
    foundit = os.path.isfile(csvfile)
    
    tofront = [
        'land_area', 
        'Simulated heliostat area', 
        'rec_width', 
        'rec_height', 
        'tht', 
        'therm_loss_base', 
        'Power incident on field', 
        'Power absorbed by the receiver', 
        'Power absorbed by HTF', 
        'q_des', 
    ]
    banlist = [
        'class_name'           , 
        'eff_file_name'        , 
        'flux_file_name'       , 
        'fluxmap_format'       , 
        'is_fluxmap_norm'      , 
        'par_save_field_img'   , 
        'par_save_flux_dat'    , 
        'par_save_flux_img'    , 
        'par_save_helio'       , 
        'par_save_summary'     , 
        'sam_grid_format'      , 
        'sam_out_dir'          , 
        'upar_save_field_img'  , 
        'upar_save_flux_dat'   , 
        'upar_save_flux_img'   , 
        'upar_save_helio'      , 
        'upar_save_summary'    , 
        'user_par_values'      , 
        'algorithm'            , 
        'aspect_display'       , 
        'class_name'           , 
        'converge_tol'         , 
        'flux_penalty'         , 
        'gs_refine_ratio'      , 
        'is_log_to_file'       , 
        'log_file_path'        , 
        'max_desc_iter'        , 
        'max_gs_iter'          , 
        'max_iter'             , 
        'max_step'             , 
        'multirec_opt_timeout' , 
        'multirec_screen_mult' , 
        'power_penalty'        , 
        'aim_method'           , 
        'cloud_depth'          , 
        'cloud_loc_x'          , 
        'cloud_loc_y'          , 
        'cloud_opacity'        , 
        'cloud_sep_depth'      , 
        'cloud_sep_width'      , 
        'cloud_shape'          , 
        'cloud_skew'           , 
        'cloud_width'          , 
        'flux_data'            , 
        'flux_day'             , 
        'flux_dist'            , 
        'flux_dni'             , 
        'flux_hour'            , 
        'flux_model'           , 
        'flux_month'           , 
        'flux_solar_az'        , 
        'flux_solar_az_in'     , 
        'flux_solar_el'        , 
        'flux_solar_el_in'     , 
        'flux_time_type'       , 
        'is_autoscale'         , 
        'is_cloud_pattern'     , 
        'is_cloud_symd'        , 
        'is_cloud_symw'        , 
        'is_cloudy'            , 
        'is_load_raydata'      , 
        'is_optical_err'       , 
        'is_save_raydata'      , 
        'is_sunshape_err'      , 
        'max_rays'             , 
        'min_rays'             , 
        'multi_rec_aim_rand'   , 
        'norm_dist_sigma'      , 
        'plot_zmax'            , 
        'plot_zmin'            , 
        'raydata_file'         , 
        'save_data'            , 
        'save_data_loc'        , 
        'seed'                 , 
        'sigma_limit_x'        , 
        'sigma_limit_y'        , 
        'x_res'                , 
        'y_res'                , 
        'class_name'           , 
        'contingency_cost'     , 
        'contingency_rate'     , 
        'fixed_cost'           , 
        'heliostat_cost'       , 
        'heliostat_spec_cost'  , 
        'is_pmt_factors'       , 
        'land_cost'            , 
        'land_spec_cost'       , 
        'pmt_factors'          , 
        'rec_cost'             , 
        'rec_cost_exp'         , 
        'rec_ref_area'         , 
        'rec_ref_cost'         , 
        'sales_tax_cost'       , 
        'sales_tax_frac'       , 
        'sales_tax_rate'       , 
        'schedule_array'       , 
        'site_cost'            , 
        'site_spec_cost'       , 
        'total_direct_cost'    , 
        'total_indirect_cost'  , 
        'total_installed_cost' , 
        'tower_cost'           , 
        'tower_exp'            , 
        'tower_fixed_cost'     , 
        'wiring_cost'          , 
        'wiring_user_spec'     , 
        'loc_city'             , 
        'loc_state'            , 
        'longitude'            , 
        'sim_time_step'        , 
        'sun_csr'              , 
        'sun_csr_adj'          , 
        'sun_pos_map'          , 
        'sun_rad_limit'        , 
        'sun_type'             , 
        'time_zone'            , 
        'user_sun'             , 
        'weather_file'         , 
        'is_multirec_powfrac'  , 
        'is_opt_zoning'        , 
        'is_prox_filter'       , 
        'is_sliprow_skipped'   , 
        'is_tht_opt'           , 
        'cant_day'             , 
        'cant_hour'            , 
        'cant_mag_i'           , 
        'cant_mag_j'           , 
        'cant_mag_k'           , 
        'cant_method'          , 
        'cant_norm_i'          , 
        'cant_norm_j'          , 
        'cant_norm_k'          , 
        'cant_rad_scaled'      , 
        'cant_radius'          , 
        'cant_sun_az'          , 
        'cant_sun_el'          , 
        'cant_vect_i'          , 
        'cant_vect_j'          , 
        'cant_vect_k'          , 
        'cant_vect_scale'      , 
        'id'                   , 
        'is_cant_rad_scaled'   , 
        'is_cant_vect_slant'   , 
        'is_enabled'           , 
        'is_faceted'           , 
        'is_focal_equal'       , 
        'is_round'             , 
        'is_xfocus'            , 
        'is_yfocus'            , 
    ] 

    for key in banlist: 
        if key in ordered: ordered.remove(key)
    for key in tofront: 
        if key in ordered: 
            ordered.remove(key)
            ordered.insert(0, key)

    with open(csvfile, 'a', newline='') as file: 
        writer = csv.writer(file)
        if foundit: 
            writer.writerow([results[key] for key in ordered])
        else: 
            writer.writerow([key for key in ordered])
            writer.writerow([results[key] for key in ordered])
def setStudyVariables(params, Preq=150, Htow=200, Hrec=50, Wrec=20, factorlosses=True): 
    # falling-particle receiver loss by power and area
    def receiverLosses(Power, Area): 
        # RMSE = 0.012
        a0 =  2.771e-02
        a1 =  5.245e-04
        a2 =  6.403e-08
        a3 = -5.735e-07

        x = Area    # [m^2]
        y = Power   # [MWt]

        losses = a1*x**1 + a2*x**2 + a3*x*y
        if losses >= 0 and losses <= 1: 
            return losses + a0
        elif losses < 0: 
            return a0
        else: 
            return 1
    # receiver
    des_par_rec = params.get(category='rec')
    des_par_rec["rec_height"]  = Hrec
    des_par_rec["rec_width"]   = Wrec
    floss = receiverLosses(Preq, Hrec*Wrec) * factorlosses
    des_par_rec["therm_loss_base"] = 1e3 * floss * Preq / (Hrec * Wrec)
    params.update(des_par_rec)
    # field
    des_par_fld = params.get(category='fld')
    des_par_fld["tht"]   = Htow
    des_par_fld["q_des"] = Preq
    params.update(des_par_fld)
def setStudyParameters(params): 
    path = os.path.join(os.getcwd(), "SolarPILOT API", "climates")
    file = "USA CA Daggett (TMY2).csv"
    # ambient
    des_par_amb = params.get(category='amb')
    des_par_amb["weather_file"] = os.path.join(path, file)
    params.update(des_par_amb)

    # heliostat
    des_par_hel = params.get(category='hel')
    des_par_hel["height"] = 12
    des_par_hel["width"]  = 12
    params.update(des_par_hel)

    # receiver
    des_par_rec = params.get(category='rec')
    des_par_rec["absorptance"] = 1.0
    des_par_rec["aperture_type"] = "Rectangular"
    des_par_rec["peak_flux"] = 10000
    des_par_rec["piping_loss"] = 0
    des_par_rec["rec_type"] = "Flat plate"
    des_par_rec["therm_loss_load"] = 0
    params.update(des_par_rec)

    # land
    des_par_lnd = params.get(category='lnd')
    des_par_lnd["max_fixed_rad"] = 5000
    des_par_lnd["max_scaled_rad"] = 20
    params.update(des_par_lnd)

    # field
    des_par_fld = params.get(category='fld')
    des_par_fld["dni_des"] = 900
    des_par_fld["q_des"] = 200
    des_par_fld["tht"] = 190
    params.update(des_par_fld)
def runInstance(copylot, params, threads): 
    # Setting up instance, assigning parameters
    rm = copylot.data_create()
    params.assign_to_instance(rm)

    # Executing SolarPILOT Simulation
    assert copylot.generate_layout(rm, nthreads=threads)
    assert copylot.simulate(rm, nthreads=threads)
    summary = copylot.summary_results(rm, save_dict=True)
    results = params.get() | summary
    assert copylot.data_free(rm)
    return results
#-------------------------------------------------#
#-------------------------------------------------#
def runStudy(x, copylot, params, threads, Power, counter): 

    # Setting up Parameters
    # H_tower, H_receiver, W_receiver = x
    H_tower, H_receiver = x
    H_tower    = 100 * H_tower      # [m]
    H_receiver = 10 * H_receiver    # [m]
    # W_receiver = 10 * W_receiver    # [m]
    setStudyVariables(
        params=params,       # designParameters object
        Preq=Power,          # Power requirement
        Htow=H_tower,        # Tower Height
        Hrec=H_receiver,     # Receiver Height
        # Wrec=W_receiver      # Receiver Width
    )

    if not isinstance(counter, int): 
        counter.value += 1
    else: counter += 1

    # Running SolarPILOT, collecting results
    solarPILOT_out = runInstance(copylot, params, threads=threads)
    Power_total = solarPILOT_out['Power absorbed by HTF']
    A_surf_heliostats = solarPILOT_out['Simulated heliostat area']
    A_surf_land = solarPILOT_out['land_area']

    # Adding penalty if power target not reached
    if (Power_total/1000) < Power: 
        Cpen = 20e6 * (Power - (Power_total/1000))
    else: Cpen = 0

    # Calculating Total Cost
    # Crec = 37400 * H_receiver * W_receiver
    Crec = 37400 * H_receiver * 20

    Chel = (75 + 10) * A_surf_heliostats
    Clnd = 2.5 * A_surf_land * 4050
    Ctow = 157.44 * H_tower ** 1.9174
    Ctot = Crec + Chel + Clnd + Ctow
    return (Ctot + Cpen) / 1e6 # [M$]
#-------------------------------------------------#
#-------------------------------------------------#
def optimizeStudy(Power, start, threads, counter): 

    cp = CoPylot()
    dp = designParameters(cp)
    setStudyParameters(dp)

    #---Optimization Problem
    # Initial guesses
    # H_tow, H_rec, W_rec = start
    H_tow, H_rec = start

    # Decision Variables
    # variables = [H_tow, H_rec, W_rec]
    variables = [H_tow, H_rec]
    bounds = [
        (1.0, 3.0),    # [m*1e-2] Tower Height
        (0.5, 2.0),    # [m*1e-1] Receiver Height
        # (2.0, 2.0)     # [m*1e-1] Receiver Width
    ]

    # Run optimization
    objective = partial(
        runStudy, 
        copylot=cp, params=dp, threads=threads, Power=Power, counter=counter
    )

    results = opt.minimize(
        objective,          # Objective function
        variables,          # Decision Variables
        method='COBYLA',    # Algorithm
        bounds=bounds,      # DV Upper / Lower Bounds
        # jac='3-point',      # method for computing gradient
        options={         
            'tol': 0.1, 
            'maxiter': 200, 
            'rhobeg': 1.0
        }
    )

    #---Generating Optimized Layout and saving results
    H_tow_opt = results.x[0] * 100
    H_rec_opt = results.x[1] * 10
    # W_rec_opt = results.x[2] * 10
    setStudyVariables(
        params=dp, 
        Preq=Power, 
        Htow=H_tow_opt, 
        Hrec=H_rec_opt, 
        # Wrec=W_rec_opt 
    )

    optimized_instance = runInstance(cp, dp, threads=threads) | results | dp.get()
    return optimized_instance
#-------------------------------------------------#
#-------------------------------------------------#
def mplistener(q, calls, maxstudies, file): 

    def report(calls, studies, maxstudies, params): 

        sys.stdout.write(f"{'studies completed':.<20}{str(int(studies))+'/'+str(int(maxstudies)):.>20}")
        sys.stdout.write(f"\n{'total iterations':.<20}{calls:.>20}")
        sys.stdout.write(f"\n")
        sys.stdout.write(f"\n{'Last completed study'}")
        sys.stdout.write(f"\n{'-> power required':.<20}{params['q_des']:.>20.2f} [MWt]")
        sys.stdout.write(f"\n{'-> power absorbed':.<20}{params['Power absorbed by HTF']/1e3:.>20.2f} [MWt]")
        sys.stdout.write(f"\n{'-> tower height':.<20}{params['tht']:.>20.3f} [m]")
        sys.stdout.write(f"\n{'-> receiver height':.<20}{params['rec_height']:.>20.3f} [m]")
        sys.stdout.write(f"\n{'-> receiver width':.<20}{params['rec_width']:.>20.3f} [m]")
        sys.stdout.write(f"\n")
        for _ in range(9): sys.stdout.write("\033[F")
        sys.stdout.flush()

    studies = 0
    now = datetime.now()
    params = {
        'q_des': np.nan, 
        'Power absorbed by HTF': np.nan, 
        'tht': np.nan,
        'rec_height': np.nan,
        'rec_width': np.nan
    }

    print(
    f"""
CSP Reduced-Order Models for Optimal Cost
--------------------------------------------
Author: Kaleb Troyer
{now.year}-{now.month:0>2}-{now.day} {now.strftime('%H:%M:%S')}

This program aims to identify optimal
dimensions for a CSP tower, falling-particle
receiver, and solar field - given some power
requirement - by using the API for solarPILOT 
and scipy.optimize. 

Optimizer Settings: 
algorithm        {'COBYLA'}
penalty          {20} [M$/MWt]

Parallelization Settings: 
multitprocesses  {True}
max processes    {'12 / 20'}
--------------------------------------------

Initializing Optimization Routine."""
    )

    while True: 
        report(int(calls.value), studies, maxstudies, params)
        try: message = q.get(timeout=2)
        except queue.Empty: continue
        if message=='kill': 
            break 
        elif isinstance(message, dict): 
            studies += 1
            params['q_des'] = message['q_des']
            params['Power absorbed by HTF'] = message['Power absorbed by HTF']
            params['tht'] = message['tht']
            params['rec_height'] = message['rec_height']
            params['rec_width'] = message['rec_width']

            saveStudyResults(message, file)
def mpworker(arg, threads, q, i):
    Preq, guess = arg

    results = optimizeStudy(
        Power=Preq, 
        start=guess, 
        threads=threads, 
        counter=i
    )

    q.put(results)
    return results
def multiProcess(cpu=4, threads=1, powers=[], guesses=[], savefile='test.csv'): 

    if cpu <= 1: 
        raise ValueError("Process requires at least two cores.")
    combinations = [(Preq, guess) for i, (Preq, guess) in enumerate([(p, g) for p in powers for g in guesses])]
    alljobs = []

    manager = mp.Manager()
    mpqueue = manager.Queue()
    tracker = manager.Value('i', 0)
    cpupool = mp.Pool(processes=cpu)
    watcher = cpupool.apply_async(mplistener, (mpqueue, tracker, len(combinations), savefile))

    for permutation in combinations: 
        job = cpupool.apply_async(mpworker, (permutation, threads, mpqueue, tracker))
        alljobs.append(job)

    for job in alljobs: 
        job.get()

    mpqueue.put('kill')
    watcher.get()
    cpupool.close()
    cpupool.join()

if __name__=='__main__': 

    multiprocessing = True
    date = datetime.now()

    cores = 8
    threads = 2
    powerToFluid = np.linspace(100, 600, 21)
    initialGuess = [
        # (Htow, Hrec, Wrec) for i, (Htow, Hrec, Wrec) in enumerate(
        (Htow, Hrec) for i, (Htow, Hrec) in enumerate(
            [
                # (H, h, w) 
                (H, h) 
                for H in [1.2, 2.0, 2.8] # [m*1e-2] Tower Height guesses
                for h in [1.0, 1.5, 2.0] # [m*1e-1] Receiver Height guesses
                # for w in [1.0, 1.8, 2.5] # [m*1e-1] Receiver Width guesses
            ]
        )
    ]

    if multiprocessing: 
        multiProcess(
            cpu=cores, threads=threads, powers=powerToFluid, guesses=initialGuess, 
            savefile=f"{date.year}-{date.month:0>2}-{date.day}_reduced-order-models.csv"
        )
    else: testCase(save=True)

    for _ in range(9): sys.stdout.write("\n")
    print("\nAll studies successfully completed.")
    print(f"Time elapsed: {datetime.now()-date}\n")

combinations = [(Power, guess) for i, (Power, guess) in enumerate(...)]
alljobs = []    

manager = mp.Manager()         # creating the process manager
mpqueue = manager.Queue()      # initializing the process queue
cpupool = mp.Pool(processes=6) # creating the pool of CPUs

for combo in combinations: # creating all jobs
    job = cpupool.apply_async(mpworker, (combo, threads, mpqueue))
    alljobs.append(job)
for job in alljobs: # sends each job to the manager, which assigns it to a process 
    job.get()

cpupool.close() # prevents any more tasks from being submitted to the pool 
cpupool.join()  # ensures all tasks have been completed before moving on
