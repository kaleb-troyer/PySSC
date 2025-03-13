import csv

class designParameters():

    def __init__(self, cp) -> None:
        self.cp = cp

        NUMBER = 0
        STRING = ''
        ARRAY  = []
        EMPTY  = None
        LOGIC  = True

        # all parameters by category, type
        self.des_par_amb = {
            'atm_atten_est' : NUMBER, # %         Average attenuation
            'atm_model'     : STRING, # none      Atmospheric attenuation model
            'class_name'    : STRING, # none      Class name
            'del_h2o'       : NUMBER, # mm        H2O Atmospheric precipitable water
            'dni_layout'    : NUMBER, # W/m2      Constant DNI for layout calculations
            'dpres'         : NUMBER, # atm       Ambient pressure
            'elevation'     : NUMBER, # m         Plant elevation
            'insol_type'    : STRING, # none      Insolation model
            'latitude'      : NUMBER, # deg       Plant latitude
            'loc_city'      : STRING, # none      Weather file location name
            'loc_state'     : STRING, # none      Weather file state name
            'longitude'     : NUMBER, # deg       Plant longitude
            'sim_time_step' : NUMBER, # sec       Simulation wea ther data time step
            'sun_csr'       : NUMBER, # none      Circumsolar ratio
            'sun_csr_adj'   : NUMBER, # none      Corrected circumsolar ratio
            'sun_pos_map'   : ARRAY,  # [deg'deg] sun_pos_map
            'sun_rad_limit' : NUMBER, # mrad      Sunshape angular extent
            'sun_type'      : STRING, # none      Sunshape model
            'time_zone'     : NUMBER, # hr        Time zone
            'user_sun'      : ARRAY,  # [deg'deg] user_sun
            'weather_file'  : STRING, # none      Weather file
        } 
        self.des_par_fin = {
            'class_name'           : STRING, # none   Class name
            'contingency_cost'     : NUMBER, # $      Contingency cost
            'contingency_rate'     : NUMBER, # %      Contingency
            'fixed_cost'           : NUMBER, # $      Fixed cost
            'heliostat_cost'       : NUMBER, # $      Heliostat field cost
            'heliostat_spec_cost'  : NUMBER, # $/m2   Heliostat field
            'is_pmt_factors'       : LOGIC,  # none   Enable payment weighting factors
            'land_cost'            : NUMBER, # $      Land cost
            'land_spec_cost'       : NUMBER, # $/acre Land cost per acre
            'pmt_factors'          : ARRAY,  # none   Payment allocation factors
            'rec_cost'             : NUMBER, # $      Receiver cost
            'rec_cost_exp'         : NUMBER, # none   Receiver cost scaling exponent
            'rec_ref_area'         : NUMBER, # m2     Receiver reference area
            'rec_ref_cost'         : NUMBER, # $      Receiver reference cost
            'sales_tax_cost'       : NUMBER, # $      Sales tax cost
            'sales_tax_frac'       : NUMBER, # %      Sales tax rate portion
            'sales_tax_rate'       : NUMBER, # %      Sales tax rate
            'schedule_array'       : ARRAY,  # none   Time series schedule of TOU periods
            'site_cost'            : NUMBER, # $      Site improvements cost
            'site_spec_cost'       : NUMBER, # $/m2   Site improvements
            'total_direct_cost'    : NUMBER, # $      Total direct cost
            'total_indirect_cost'  : NUMBER, # $      Total indirect cost
            'total_installed_cost' : NUMBER, # $      Total installed cost
            'tower_cost'           : NUMBER, # $      Tower cost
            'tower_exp'            : NUMBER, # none   Tower cost scaling exponent
            'tower_fixed_cost'     : NUMBER, # $      Fixed tower cost
            'wiring_cost'          : NUMBER, # $      Wiring cost
            'wiring_user_spec'     : NUMBER, # $/m2   Wiring specific cost
        }
        self.des_par_sim = {
            'aim_method'         : STRING, # none   Heliostat aim point method
            'class_name'         : STRING, # none   Class name
            'cloud_depth'        : NUMBER, # m      Cloud depth
            'cloud_loc_x'        : NUMBER, # m      Cloud location (X)
            'cloud_loc_y'        : NUMBER, # m      Cloud location (Y)
            'cloud_opacity'      : NUMBER, # none   Cloud opacity
            'cloud_sep_depth'    : NUMBER, # none   Cloud pattern depth spacing mult.
            'cloud_sep_width'    : NUMBER, # none   Cloud pattern width spacing mult.
            'cloud_shape'        : STRING, # none   Cloud shape
            'cloud_skew'         : NUMBER, # deg    Cloud orientation angle
            'cloud_width'        : NUMBER, # m      Cloud width
            'flux_data'          : EMPTY,  # none   Flux profile in string form
            'flux_day'           : NUMBER, # day    Day of the month
            'flux_dist'          : STRING, # none   Aim point distribution sampling
            'flux_dni'           : NUMBER, # W/m2   Direct Normal Irradiation
            'flux_hour'          : NUMBER, # hr     Hour of the day
            'flux_model'         : STRING, # none   Flux simulation model
            'flux_month'         : NUMBER, # month  Month of the year
            'flux_solar_az'      : NUMBER, # deg    Calculated solar azimuth angle
            'flux_solar_az_in'   : NUMBER, # none   Solar azimuth angle (0 = N)
            'flux_solar_el'      : NUMBER, # deg    Calculated solar elevation angle
            'flux_solar_el_in'   : NUMBER, # none   Solar elevation angle
            'flux_time_type'     : STRING, # none   Simulation time spec. method
            'is_autoscale'       : LOGIC,  # none   Autoscale
            'is_cloud_pattern'   : LOGIC,  # none   Enable cloud pattern
            'is_cloud_symd'      : LOGIC,  # none   Pattern symmetric in depth direction
            'is_cloud_symw'      : LOGIC,  # none   Pattern symmetric in width direction
            'is_cloudy'          : LOGIC,  # none   Simulate cloud transient
            'is_load_raydata'    : LOGIC,  # none   Load existing heliostat ray data
            'is_optical_err'     : LOGIC,  # none   Include optical errors
            'is_save_raydata'    : LOGIC,  # none   Save heliostat stage ray data
            'is_sunshape_err'    : LOGIC,  # none   Include sun shape
            'max_rays'           : NUMBER, # none   Maximum number of generated rays
            'min_rays'           : NUMBER, # none   Desired number of ray intersections
            'multi_rec_aim_rand' : NUMBER, # none   Multi-receiver aimpoint randomization
            'norm_dist_sigma'    : NUMBER, # none   Aiming distribution standard dev.
            'plot_zmax'          : NUMBER, # none   Max
            'plot_zmin'          : NUMBER, # none   Min
            'raydata_file'       : STRING, # none   Ray data file
            'save_data'          : LOGIC,  # none   Save all raytrace data
            'save_data_loc'      : STRING, # none   ray data
            'seed'               : NUMBER, # none   Seed value (-1 for automatic selection)
            'sigma_limit_x'      : NUMBER, # none   Min. image offset from receiver edge - X
            'sigma_limit_y'      : NUMBER, # none   Min. image offset from receiver edge - Y
            'x_res'              : NUMBER, # none   Flux grid resolution - Horizontal
            'y_res'              : NUMBER, # none   Flux grid resolution - Vertical
        } 
        self.des_par_hel = {
            'area'               : NUMBER, # m2     Total reflective aperture area
            'cant_day'           : NUMBER, # day    Canting day of the year
            'cant_hour'          : NUMBER, # hr     Canting hour (past noon)
            'cant_mag_i'         : NUMBER, # none   Total canting vector - i
            'cant_mag_j'         : NUMBER, # none   Total canting vector - j
            'cant_mag_k'         : NUMBER, # none   Total canting vector - k
            'cant_method'        : STRING, # none   Heliostat canting method
            'cant_norm_i'        : NUMBER, # none   Normalized canting vector - i
            'cant_norm_j'        : NUMBER, # none   Normalized canting vector - j
            'cant_norm_k'        : NUMBER, # none   Normalized canting vector - k
            'cant_rad_scaled'    : NUMBER, # none   Canting radius factor
            'cant_radius'        : NUMBER, # m      Canting radius
            'cant_sun_az'        : NUMBER, # deg    Sun azimuth at canting
            'cant_sun_el'        : NUMBER, # deg    Sun elevation at canting
            'cant_vect_i'        : NUMBER, # none   Canting vector x-component
            'cant_vect_j'        : NUMBER, # none   Canting vector y-component
            'cant_vect_k'        : NUMBER, # none   Canting vector z-component
            'cant_vect_scale'    : NUMBER, # m      Canting vector magnitude
            'cbdata'             : EMPTY,  # none   Data pointer for UI page
            'class_name'         : STRING, # none   Class name
            'diameter'           : NUMBER, # m      Heliostat diameter
            'err_azimuth'        : NUMBER, # rad    Azimuth pointing error
            'err_elevation'      : NUMBER, # rad    Elevation pointing error
            'err_reflect_x'      : NUMBER, # rad    Reflected beam error in X
            'err_reflect_y'      : NUMBER, # rad    Reflected beam error in Y
            'err_surface_x'      : NUMBER, # rad    Surface slope error in X
            'err_surface_y'      : NUMBER, # rad    Surface slope error in Y
            'err_total'          : NUMBER, # rad    Total reflected image error
            'focus_method'       : STRING, # none   Heliostat focusing type
            'height'             : NUMBER, # m      Structure height
            'helio_name'         : STRING, # none   Heliostat template name
            'id'                 : NUMBER, # none   id
            'is_cant_rad_scaled' : LOGIC,  # none   Scale cant radius with tower height
            'is_cant_vect_slant' : LOGIC,  # none   Scale vector with slant range
            'is_enabled'         : LOGIC,  # none   Is template enabled?
            'is_faceted'         : LOGIC,  # none   Use multiple panels
            'is_focal_equal'     : LOGIC,  # none   Use single focal length
            'is_round'           : STRING, # none   Heliostat shape
            'is_xfocus'          : LOGIC,  # none   Focus in X
            'is_yfocus'          : LOGIC,  # none   Focus in Y
            'n_cant_x'           : NUMBER, # none   No. horizontal panels
            'n_cant_y'           : NUMBER, # none   No. vertical panels
            'r_collision'        : NUMBER, # m      Heliostat collision radius
            'ref_total'          : NUMBER, # none   Total optical reflectance
            'reflect_ratio'      : NUMBER, # none   Reflective surface ratio
            'reflectivity'       : NUMBER, # none   Mirror reflectivity
            'rvel_max_x'         : NUMBER, # rad/s  rvel_max_x
            'rvel_max_y'         : NUMBER, # rad/s  rvel_max_y
            'soiling'            : NUMBER, # none   Soiling factor
            'st_err_type'        : STRING, # none   Optical error type (SolTrace only)
            'temp_az_max'        : NUMBER, # deg    Max. angular boundary for heliostat type
            'temp_az_min'        : NUMBER, # deg    Min. angular boundary for heliostat type
            'temp_rad_max'       : NUMBER, # none   Maximum radius for heliostat type
            'temp_rad_min'       : NUMBER, # none   Minimum radius for heliostat type
            'template_order'     : NUMBER, # none   template_order
            'track_method'       : STRING, # none   Heliostat tracking update method
            'track_period'       : NUMBER, # sec    Heliostat tracking update period
            'type'               : NUMBER, # none   type
            'width'              : NUMBER, # m      Structure width
            'x_focal_length'     : NUMBER, # m      Focal length in X
            'x_gap'              : NUMBER, # m      Cant panel horiz. gap
            'y_focal_length'     : NUMBER, # m      Focal length in Y
            'y_gap'              : NUMBER, # m      Cant panel vert. gap
        } 
        self.des_par_lnd = {
            'bound_area'             : NUMBER, # acre   Solar field land area
            'class_name'             : STRING, # none   Class name
            'exclusions'             : EMPTY,  # none   exclusions
            'import_tower_lat'       : NUMBER, # deg    Imported land boundary tower latitude
            'import_tower_lon'       : NUMBER, # deg    Imported land boundary tower longitude
            'import_tower_set'       : LOGIC,  # none   Imported land boundary tower flag
            'inclusions'             : EMPTY,  # none   inclusions
            'is_bounds_array'        : LOGIC,  # none   Use land boundary array
            'is_bounds_fixed'        : LOGIC,  # none   Use fixed land bounds
            'is_bounds_scaled'       : LOGIC,  # none   Bounds scale with tower height
            'is_exclusions_relative' : LOGIC,  # none   Exclusions relative to tower position
            'land_area'              : NUMBER, # acre   Total land area
            'land_const'             : NUMBER, # acre   Non-solar field land area
            'land_mult'              : NUMBER, # none   Solar field land area multiplier
            'max_fixed_rad'          : NUMBER, # m      Maximum land radius (fixed)
            'max_scaled_rad'         : NUMBER, # none   Maximum field radius
            'min_fixed_rad'          : NUMBER, # m      Minimum land radius (fixed)
            'min_scaled_rad'         : NUMBER, # none   Minimum field radius
            'radmax_m'               : NUMBER, # m      Maximum heliostat distance
            'radmin_m'               : NUMBER, # m      Minimum heliostat distance
            'tower_offset_x'         : NUMBER, # m      Tower location offset - X
            'tower_offset_y'         : NUMBER, # m      Tower location offset - Y
        }
        self.des_par_opt = {
            'algorithm'            : STRING, # none   Optimization algorithm
            'aspect_display'       : NUMBER, # none   Current receiver aspect ratio (H/W)
            'class_name'           : STRING, # none   Class name
            'converge_tol'         : NUMBER, # none   Convergence tolerance
            'flux_penalty'         : NUMBER, # none   Flux overage penalty
            'gs_refine_ratio'      : NUMBER, # none   Refinement relative bounding box
            'is_log_to_file'       : LOGIC,  # none   Echo log to file
            'log_file_path'        : STRING, # none   Log file location
            'max_desc_iter'        : NUMBER, # none   Max. no. continuous descent steps
            'max_gs_iter'          : NUMBER, # none   Max. refinement iterations
            'max_iter'             : NUMBER, # none   Maximum iterations
            'max_step'             : NUMBER, # none   Initial step size
            'multirec_opt_timeout' : NUMBER, # sec    Multi-receiver optimization solver time
            'multirec_screen_mult' : NUMBER, # none   Multi-receiver heliostat screen fraction
            'power_penalty'        : NUMBER, # none   Power shortage penalty
        }
        self.des_par_par = {
            'class_name'          : STRING, # none   Class name
            'eff_file_name'       : STRING, # none   Efficiency file name
            'flux_file_name'      : STRING, # none   Fluxmap file name
            'fluxmap_format'      : STRING, # none   Fluxmap data dimensions
            'is_fluxmap_norm'     : LOGIC,  # none   Normalize flux data
            'par_save_field_img'  : LOGIC,  # none   Save field image
            'par_save_flux_dat'   : LOGIC,  # none   Save receiver flux data
            'par_save_flux_img'   : LOGIC,  # none   Save receiver flux image
            'par_save_helio'      : LOGIC,  # none   Save heliostat performance data
            'par_save_summary'    : LOGIC,  # none   Save performance summary information
            'sam_grid_format'     : STRING, # none   SAM data grid format
            'sam_out_dir'         : EMPTY,  # none   Output directory
            'upar_save_field_img' : LOGIC,  # none   Save field image (User)
            'upar_save_flux_dat'  : LOGIC,  # none   Save receiver flux data (User)
            'upar_save_flux_img'  : LOGIC,  # none   Save receiver flux image (User)
            'upar_save_helio'     : LOGIC,  # none   Save heliostat performance data (User)
            'upar_save_summary'   : LOGIC,  # none   Save performance summary info (User)
            'user_par_values'     : EMPTY,  # none   User parametric values
        }
        self.des_par_rec = {
            'absorber_area'        : NUMBER, # m2     Receiver absorber area
            'absorptance'          : NUMBER, # none   Receiver thermal absorptance
            'accept_ang_type'      : STRING, # none   Receiver acceptance angles shape
            'accept_ang_x'         : NUMBER, # deg    Receiver horizontal acceptance angle
            'accept_ang_y'         : NUMBER, # deg    Receiver vertical acceptance angle
            'aperture_area'        : NUMBER, # m2     Receiver aperture area
            'aperture_type'        : STRING, # none   Aperture geometry shape
            'cbdata'               : EMPTY,  # none   Data pointer for UI page
            'class_name'           : STRING, # none   Class name
            'flux_profile_type'    : STRING, # none   Desired receiver flux profile
            'id'                   : NUMBER, # none   Template ID
            'is_aspect_opt'        : LOGIC,  # none   Optimize receiver aspect ratio
            'is_enabled'           : LOGIC,  # none   Is template enabled?
            'is_open_geom'         : LOGIC,  # none   Limit receiver panel span angle
            'is_polygon'           : LOGIC,  # none   Represent receiver as polygon
            'map_color'            : STRING, # none   Specified receiver map color
            'n_panels'             : NUMBER, # none   Number of receiver panels
            'n_user_flux_profile'  : NUMBER, # none   Normalized user flux profile
            'optical_height'       : NUMBER, # m      Receiver optical height
            'panel_rotation'       : NUMBER, # deg    Receiver panel azimuthal orientation
            'peak_flux'            : NUMBER, # kW/m2  Allowable peak flux
            'piping_loss'          : NUMBER, # MW     Receiver piping loss
            'q_rec_des'            : NUMBER, # MW     Design-point thermal power
            'piping_loss_coef'     : NUMBER, # kW/m   Receiver piping loss coefficient
            'rec_azimuth'          : NUMBER, # deg    Receiver orientation azimuth
            'piping_loss_const'    : NUMBER, # kW     Receiver piping loss constant
            'rec_cav_apw'          : NUMBER, # m      Cavity aperture width
            'power_fraction'       : NUMBER, # none   Multi-receiver power fraction
            'rec_cav_blip'         : NUMBER, # none   Cavity bottom lip fractional height
            'rec_aspect'           : NUMBER, # none   Receiver aspect ratio (H/W)
            'rec_cav_rad'          : NUMBER, # m      Cavity active surface radius
            'rec_cav_aph'          : NUMBER, # m      Cavity aperture height
            'rec_diameter'         : NUMBER, # m      Receiver diameter
            'rec_cav_apwfrac'      : NUMBER, # none   Cavity aperture width fraction
            'rec_height'           : NUMBER, # m      Receiver height
            'rec_cav_cdepth'       : NUMBER, # none   Cavity absorber centroid aperture offset
            'rec_offset_reference' : STRING, # none   Receiver offset reference
            'rec_cav_tlip'         : NUMBER, # none   Cavity top lip fractional height
            'rec_offset_x_global'  : NUMBER, # m      Receiver global positioning offset - X axis
            'rec_elevation'        : NUMBER, # deg    Receiver orientation elevation
            'rec_offset_y_global'  : NUMBER, # m      Receiver global positioning offset - Y axis
            'rec_name'             : STRING, # none   Receiver template name
            'rec_offset_z_global'  : NUMBER, # m      Receiver global positioning offset - Z axis
            'rec_offset_x'         : NUMBER, # m      Receiver positioning offset - X axis
            'rec_width'            : NUMBER, # m      Receiver width
            'rec_offset_y'         : NUMBER, # m      Receiver positioning offset - Y axis
            'rec_offset_z'         : NUMBER, # m      Receiver positioning offset - Z axis
            'rec_type'             : STRING, # none   Receiver type
            'therm_eff'            : NUMBER, # none   Receiver calculated thermal efficiency
            'therm_loss'           : NUMBER, # MW     Design-point thermal loss
            'therm_loss_base'      : NUMBER, # kW/m2  Design point receiver thermal loss
            'therm_loss_load'      : ARRAY,  # none   Load-based thermal loss adjustment
            'therm_loss_wind'      : ARRAY,  # none   Wind-based thermal loss adjustment
        } 
        self.des_par_fld = {
            'accept_max'          : NUMBER, # deg   Maximum solar field extent angle
            'accept_min'          : NUMBER, # deg   Minimum solar field extent angle
            'az_spacing'          : NUMBER, # none  Azimuthal spacing factor
            'class_name'          : STRING, # none  Class name
            'des_sim_detail'      : STRING, # none  Optimization simulations
            'des_sim_ndays'       : NUMBER, # none  Number of days to simulate
            'des_sim_nhours'      : NUMBER, # none  Simulation hour frequency
            'dni_des'             : NUMBER, # W/m2  Design-point DNI value
            'hsort_method'        : STRING, # none  Heliostat selection criteria
            'interaction_limit'   : NUMBER, # none  helio-ht Heliostat shading interaction limit
            'is_multirec_powfrac' : LOGIC,  # none  Specify multi-receiver power fractions
            'is_opt_zoning'       : LOGIC,  # none  Enable optical layout zone method
            'is_prox_filter'      : LOGIC,  # none  Apply proximity filter
            'is_sliprow_skipped'  : LOGIC,  # none  Offset slip plane for blocking
            'is_tht_opt'          : LOGIC,  # none  Optimize tower height
            'layout_method'       : STRING, # none  Layout method
            'max_zone_size_az'    : NUMBER, # none  tower-ht Max. optical layout zone size - azimuthal
            'max_zone_size_rad'   : NUMBER, # none  tower-ht Max. optical layout zone size - radial
            'min_zone_size_az'    : NUMBER, # none  tower-ht Min. optical layout zone size - azimuthal
            'min_zone_size_rad'   : NUMBER, # none  tower-ht Min. optical layout zone size - radial
            'prox_filter_frac'    : NUMBER, # none  Proximi ty filter fraction
            'q_des'               : NUMBER, # MWt   Solar field design power
            'rad_spacing_method'  : STRING, # none  Radial spacing method
            'rec_area'            : NUMBER, # m2    Total receiver surface area
            'row_spacing_x'       : NUMBER, # none  Heliostat spacing factor - X direction
            'row_spacing_y'       : NUMBER, # none  Heliostat spacing factor - Y direction
            'sf_area'             : NUMBER, # m2    Total solar field area
            'shadow_height'       : NUMBER, # m     Tower shadow height
            'shadow_width'        : NUMBER, # m     Tower shadow width
            'slip_plane_blocking' : NUMBER, # none  Allowable blocking in slip plane
            'spacing_reset'       : NUMBER, # none  Azimuthal spacing reset limit
            'sun_az_des'          : NUMBER, # deg   Calculated design-point solar azimuth
            'sun_az_des_user'     : NUMBER, # deg   Specified design-point solar azimuth
            'sun_el_des'          : NUMBER, # deg   Calculated design-point solar elevation
            'sun_el_des_user'     : NUMBER, # deg   Specified design-point solar elevation
            'sun_loc_des'         : STRING, # none  Sun location at design point
            'template_rule'       : STRING, # none  Heliostat geometry distribution
            'tht'                 : NUMBER, # m     Tower optical height
            'trans_limit_fact'    : NUMBER, # none  Packing transition limit factor
            'xy_field_shape'      : STRING, # none  Heliostat field layout shape
            'xy_rect_aspect'      : NUMBER, # none  Layout aspect ratio (Y/X)
            'zone_div_tol'        : NUMBER, # none  Optical layout zone mesh tolerance
        }

        # all parameter categories and metadata
        self.categories = [
            (self.des_par_amb, 'amb', 'ambient'),
            (self.des_par_fin, 'fin', 'financial'),
            (self.des_par_sim, 'sim', 'fluxsim'),
            (self.des_par_hel, 'hel', 'heliostat'),
            (self.des_par_lnd, 'lnd', 'land'),
            (self.des_par_opt, 'opt', 'optimize'),
            (self.des_par_par, 'par', 'parametric'),
            (self.des_par_rec, 'rec', 'receiver'),
            (self.des_par_fld, 'fld', 'solarfield'),
        ]

        # getting parameter defaults from temporary instance
        self.instance = self.cp.data_create()
        self.__data_get_all()
        self.cp.data_free(self.instance)
        self.instance = None

    def assign_to_instance(self, instance) -> None:
        self.instance = instance
        self.__data_set_all()
        self.instance = None
    #---Get Current Design Parameters
    def get(self, field: str = '', category: str = '') -> dict:
        def retrieve(parameters, field):
            return parameters[field] if field else parameters.copy()
        if self.instance != None: self.__data_get_all()
        if category:
            for item in self.categories:
                if set(item[1]).issubset(set(category)):
                    return retrieve(item[0], field)
            raise KeyError(f"Category not found.")
        else:
            des_par_all = {}
            for item in self.categories:
                des_par_all.update(item[0])
            return retrieve(des_par_all, field)
    def to_csv(self, name : str = 'parameters.csv', path : str = ''):
        des_par_all = {}
        for category in self.categories:
            cat_dict = category[0]
            cat_name = category[2]
            for key, value in cat_dict.items():
                des_par_all[cat_name+".0."+key] = value
        output = path+name
        with open(output, mode='w', newline='') as file:
            writer = csv.writer(file)
            for key, value in des_par_all.items():
                writer.writerow([key, value])
    def category_info(self):
        print("symbol\t   category")
        print(25*"-")
        for category in self.categories:
            print(f"{category[1]}\t-> {category[2]}")
    #---Set New Design Parameters
    def update(self, parameters : dict, category : str = '') -> None:
        if not category:
            key = next(iter(parameters))
            for item in self.categories:
                if key in item[0].keys():
                    item[0].update(parameters)
                    return None
        else:
            for item in self.categories:
                if set(item[1]).issubset(set(category)):
                    item[0].update(parameters)
                    return None
        raise KeyError("No matching category or keys found. Please specify a valid category.")
    #---Hidden Get/Set Routines
    def __data_get_type(self, value) -> int:
        if isinstance(value, (float, bool, int)):
            return 0 # Number
        elif isinstance(value, list) or isinstance(value, str) and "'" in value:
            return 1 # Array
        elif isinstance(value, str):
            return 2 # String
        else:
            return 3 # NoneType
    def __data_set_parameter(self, parameters, category) -> None:
        field = category+".0."
        for key, value in parameters.items():
            match self.__data_get_type(value):
                case 0:
                    self.cp.data_set_number(self.instance, field+key, value)
                case 1:
                    self.cp.data_set_array(self.instance, field+key, value)
                case 2:
                    self.cp.data_set_string(self.instance, field+key, value)
                case 3:
                    pass
    def __data_get_parameter(self, parameters, category) -> None:
        category = category+".0."
        for key, value in parameters.items():
            field = category+key
            match self.__data_get_type(value):
                case 0:
                    parameters[key] = self.cp.data_get_number(
                        self.instance, field
                    )
                case 1:
                    parameters[key] = self.cp.data_get_array(
                        self.instance, field
                    )
                case 2:
                    parameters[key] = self.cp.data_get_string(
                        self.instance, field
                    )
                case 3:
                    parameters[key] = None
    def __data_get_all(self) -> None:
        for category in self.categories:
            cat_dict = category[0]
            cat_name = category[2]
            self.__data_get_parameter(cat_dict, cat_name)
    def __data_set_all(self) -> None:
        for category in self.categories:
            cat_dict = category[0]
            cat_name = category[2]
            self.__data_set_parameter(cat_dict, cat_name)

