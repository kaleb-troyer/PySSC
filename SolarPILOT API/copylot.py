import sys, os
import pandas as pd
from ctypes import *
c_number = c_double   #must be either c_double or c_float depending on copilot.h definition
import pysoltrace
import math

@CFUNCTYPE(c_int, c_number, c_char_p)
def api_callback(fprogress, msg):
    """Callback function for API -> prints message from SolarPILOT DLL"""
    newline = False
    if fprogress != 0:
        print("\rProgress is {:.2f} %".format(fprogress*100), end = "\r")
        newline = True
    if msg.decode() != '':
        if newline:
            print("\n")
        print("C++ API message -> {:s}".format(msg.decode()))
    return 1

class CoPylot:
    """
    A class to access CoPylot (SolarPILOT's Python API)

    Attributes
    ----------
    pdll : class ctypes.CDLL
        loaded SolarPILOT library of exported functions

    Methods
    -------
    version(p_data: int) -> str
        Provides SolarPILOT version number
    data_create() -> int
        Creates an instance of SolarPILOT in memory
    data_free(p_data: int) -> bool
        Frees SolarPILOT instance from memory
    api_callback_create(p_data: int) -> None
        Creates a callback function for message transfer
    api_disable_callback(p_data: int) -> None
        Disables callback function
    data_set_number(p_data: int, name: str, value) -> bool
        Sets a SolarPILOT numerical variable, used for float, int, bool, and numerical combo options.
    data_set_string(p_data: int, name: str, svalue: str) -> bool
        Sets a SolarPILOT string variable, used for string and combos
    data_set_array(p_data: int, name: str, parr: list) -> bool
        Sets a SolarPILOT array variable, used for double and int vectors
    data_set_array_from_csv(p_data: int, name: str, fn: str) -> bool
        Sets a SolarPILOT vector variable from a csv, used for double and int vectors
    data_set_matrix(p_data: int, name: str, mat: list) -> bool
        Sets a SolarPILOT matrix variable, used for double and int matrix
    data_set_matrix_from_csv(p_data: int, name: str, fn: str) -> bool
        Sets a SolarPILOT matrix variable from a csv, used for double and int matrix
    data_get_number(p_data: int, name: str) -> float
        Gets a SolarPILOT numerical variable value
    data_get_string(p_data: int, name: str) -> str
        Gets a SolarPILOT string variable value
    data_get_array(p_data: int, name: str) -> list
        Gets a SolarPILOT array (vector) variable value
    data_get_matrix(p_data: int,name: str) -> list
        Gets a SolarPILOT matrix variable value
    reset_vars(p_data: int) -> bool
        Resets SolarPILOT variable values to defaults
    add_receiver(p_data: int, rec_name: str) -> int
        Creates a receiver object
    drop_receiver(p_data: int, rec_name: str) -> bool
        Deletes a receiver object
    add_heliostat_template(p_data: int, helio_name: str) -> int
        Creates a heliostat template object
    drop_heliostat_template(p_data: int, helio_name: str) -> bool
        Deletes heliostat template object
    update_geometry(p_data: int) -> bool
        Refresh the solar field, receiver, or ambient condition settings based on current parameter settings
    generate_layout(p_data: int, nthreads: int = 0) -> bool
        Create a solar field layout
    assign_layout(p_data: int, helio_data: list, nthreads: int = 0) -> bool
        Run layout with specified positions, (optional canting and aimpoints)
    get_layout_info(p_data: int, get_corners: bool = False, restype: str = "dataframe")
        Get information regarding the heliostat field layout
    simulate(p_data: int, nthreads: int = 1, update_aimpoints: bool = True) -> bool
        Calculate heliostat field performance
    summary_results(p_data: int, save_dict: bool = True)
        Prints table of summary results from each simulation
    detail_results(p_data: int, selhel: list = None, restype: str = "dataframe", get_corners: bool = False)
        Get heliostat field layout detail results
    get_fluxmap(p_data: int, rec_id: int = 0) -> list
        Retrieve the receiver fluxmap, optionally specifying the receiver ID to retrive
    clear_land(p_data: int, clear_type: str = None) -> None
        Reset the land boundary polygons, clearing any data
    add_land(p_data: int, add_type: str, poly_points: list, is_append: bool = True) -> bool
        Add land inclusion or a land exclusion region within a specified polygon
    heliostats_by_region(p_data: int, coor_sys: str = 'all', **kwargs)
        Returns heliostats that exists within a region
    modify_heliostats(p_data: int, helio_dict: dict) -> bool
        Modify attributes of a subset of heliostats in the current layout
    save_from_script(p_data: int, sp_fname: str) -> bool
        Save the current case as a SolarPILOT .spt file
    dump_varmap_tofile(p_data: int, fname: str) -> bool
        Dump the variable structure to a text csv file
    """

    def __init__(self, debug: bool = False):
        cwd = os.path.join(os.getcwd(), 'SolarPILOT API')
        is_debugging = debug
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            if is_debugging:
                self.pdll = CDLL(cwd + "/solarpilotd.dll")
                # self.pdll = CDLL("C:\\repositories\\solarpilot\\deploy\\api\\solarpilotd.dll")
            else:
                self.pdll = CDLL(cwd + "/solarpilot.dll")
        elif sys.platform == 'darwin':
            self.pdll = CDLL(cwd + "/solarpilot.dylib")  # Never tested
        elif sys.platform.startswith('linux'):
            self.pdll = CDLL(cwd +"/solarpilot.so")  # Never tested
        else:
            print( 'Platform not supported ', sys.platform)
    
    def __unitvect(self, pt : pysoltrace.Point ):
        r = (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z)**0.5
        return pysoltrace.Point( pt.x/r, pt.y/r, pt.z/r )
    
    def version(self, p_data: int) -> str:
        """Provides SolarPILOT version number
        
        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance 

        Returns
        -------
        str
            SolarPILOT version number
        """

        self.pdll.sp_version.restype = c_char_p
        return self.pdll.sp_version(c_void_p(p_data) ).decode()

    def data_create(self) -> int:
        """Creates an instance of SolarPILOT in memory

        Returns
        -------
        int
            memory address of SolarPILOT instance 
        """

        self.pdll.sp_data_create.restype = c_void_p
        return self.pdll.sp_data_create()

    def data_free(self, p_data: int) -> bool:
        """Frees SolarPILOT instance from memory

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance 

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        
        try:
            # Ensure that pdll and sp_data_free are properly initialized
            if not hasattr(self.pdll, 'sp_data_free'):
                raise RuntimeError('sp_data_free function not found in pdll')
            
            # Define the return type of the function
            self.pdll.sp_data_free.restype = c_bool
            
            # Call the function
            result = self.pdll.sp_data_free(c_void_p(p_data))
            
            return result
        
        except Exception as e:
            print(f"Error: {e}")
            return False

    def api_callback_create(self,p_data: int) -> None:
        """Creates a callback function for message transfer
        
        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        """

        self.pdll.sp_set_callback(c_void_p(p_data), api_callback)

    def api_disable_callback(self,p_data: int) -> None:
        """Disables callback function
                
        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        """

        self.pdll.sp_disable_callback(c_void_p(p_data))

    #SPEXPORT bool sp_set_number(sp_data_t p_data, const char* name, sp_number_t v);
    def data_set_number(self, p_data: int, name: str, value) -> bool:
        """Sets a SolarPILOT numerical variable, used for float, int, bool, and numerical combo options.

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name
        value: float, int, bool
            Desired variable value

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        self.pdll.sp_set_number.restype = c_bool
        return self.pdll.sp_set_number(c_void_p(p_data), c_char_p(name.encode()), c_number(value)) 

    #SPEXPORT bool sp_set_string(sp_data_t p_data, const char *name, const char *value)
    def data_set_string(self, p_data: int, name: str, svalue: str) -> bool:
        """Sets a SolarPILOT string variable, used for string and combos

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name
        svalue : str
            Desired variable str value

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        self.pdll.sp_set_string.restype = c_bool
        return self.pdll.sp_set_string(c_void_p(p_data), c_char_p(name.encode()), c_char_p(svalue.encode()))

    #SPEXPORT bool sp_set_array(sp_data_t p_data, const char *name, sp_number_t *pvalues, int length)
    def data_set_array(self, p_data: int, name: str, parr: list) -> bool:
        """Sets a SolarPILOT array variable, used for double and int vectors

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name
        parr : list
            Vector of data (float or int) \n

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        count = len(parr)
        arr = (c_number*count)()
        arr[:] = parr # set all at once
        self.pdll.sp_set_array.restype = c_bool
        return self.pdll.sp_set_array(c_void_p(p_data), c_char_p(name.encode()), pointer(arr), c_int(count))

    # Set array variable through a csv file
    def data_set_array_from_csv(self, p_data: int, name: str, fn: str) -> bool:
        """Sets a SolarPILOT vector variable from a csv, used for double and int vectors

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name
        fn : str
            CSV file path

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        f = open(fn, 'r', encoding="utf-8-sig")
        data = []
        for line in f:
            data.extend([n for n in map(float, line.split(','))])
        f.close()
        return self.data_set_array(p_data, name, data)

    #SPEXPORT bool sp_set_matrix(sp_data_t p_data, const char *name, sp_number_t *pvalues, int nrows, int ncols)
    def data_set_matrix(self, p_data: int, name: str, mat: list) -> bool:
        """Sets a SolarPILOT matrix variable, used for double and int matrix

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name
        mat : list of list
            Matrix of data

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        nrows = len(mat)
        ncols = len(mat[0])
        size = nrows*ncols
        arr = (c_number*size)()
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                arr[idx] = c_number(mat[r][c])
                idx += 1
        self.pdll.sp_set_matrix.restype = c_bool
        return self.pdll.sp_set_matrix( c_void_p(p_data), c_char_p(name.encode()), pointer(arr), c_int(nrows), c_int(ncols))

    # Set matrix variable values through a csv file
    def data_set_matrix_from_csv(self, p_data: int, name: str, fn: str) -> bool:
        """Sets a SolarPILOT matrix variable from a csv, used for double and int matrix

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name
        fn : str
            CSV file path

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        f = open(fn, 'r', encoding="utf-8-sig")
        data = [] 
        for line in f : 
            lst = ([n for n in map(float, line.split(','))])
            data.append(lst)
        f.close()
        return self.data_set_matrix(p_data, name, data) 

    #SPEXPORT sp_number_t sp_get_number(sp_data_t p_data, const char* name)
    def data_get_number(self, p_data: int, name: str) -> float:
        """Gets a SolarPILOT numerical variable value

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name

        Returns
        -------
        float
            Variable value 
        """

        self.pdll.sp_get_number.restype = c_number
        return self.pdll.sp_get_number(c_void_p(p_data), c_char_p(name.encode()))

    #SPEXPORT const char *sp_get_string(sp_data_t p_data, const char *name)
    def data_get_string(self, p_data: int, name: str) -> str:
        """Gets a SolarPILOT string variable value

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name

        Returns
        -------
        str
            Variable value 
        """

        self.pdll.sp_get_string.restype = c_char_p
        return self.pdll.sp_get_string(c_void_p(p_data), c_char_p(name.encode())).decode()

    #SPEXPORT sp_number_t *sp_get_array(sp_data_t p_data, const char *name, int *length)
    def data_get_array(self, p_data: int, name: str) -> list:
        """Gets a SolarPILOT array (vector) variable value

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name

        Returns
        -------
        list
            Variable value
        """

        count = c_int()
        self.pdll.sp_get_array.restype = POINTER(c_number)
        parr = self.pdll.sp_get_array(c_void_p(p_data), c_char_p(name.encode()), byref(count))
        arr = parr[0:count.value]
        return arr

    #SPEXPORT sp_number_t *sp_get_matrix(sp_data_t p_data, const char *name, int *nrows, int *ncols)
    def data_get_matrix(self,p_data: int,name: str) -> list:
        """Gets a SolarPILOT matrix variable value

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        name : str
            SolarPILOT variable name

        Returns
        -------
        list of list
            Variable value 
        """

        nrows = c_int()
        ncols = c_int()
        self.pdll.sp_get_matrix.restype = POINTER(c_number)
        parr = self.pdll.sp_get_matrix( c_void_p(p_data), c_char_p(name.encode()), byref(nrows), byref(ncols) )
        mat = []
        for r in range(nrows.value):
            row = []
            for c in range(ncols.value):
                row.append( float(parr[ncols.value * r + c]))
            mat.append(row)
        return mat

    #SPEXPORT void sp_reset_geometry(sp_data_t p_data)
    def reset_vars(self, p_data: int) -> bool:
        """Resets SolarPILOT variable values to defaults

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance

        Returns
        -------
        bool
            True if successful, False otherwise 
        """

        return self.pdll.sp_reset_geometry( c_void_p(p_data))

    #SPEXPORT int sp_add_receiver(sp_data_t p_data, const char* receiver_name)
    def add_receiver(self, p_data: int, rec_name: str) -> int:
        """Creates a receiver object

        NOTE: CoPylot starts with a default receiver configuration at receiver object ID = 0, with 'Receiver 1' as the receiver's name. 
        If you add a receiver object without dropping this default receiver, generating a layout will result in a multi-receiver problem, 
        which could produce strange results.

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        rec_name : str
            Receiver name

        Returns
        -------
        int
            Receiver object ID 
        """

        self.pdll.sp_add_receiver.restype = c_int
        return self.pdll.sp_add_receiver( c_void_p(p_data), c_char_p(rec_name.encode()))

    #SPEXPORT bool sp_drop_receiver(sp_data_t p_data, const char* receiver_name)
    def drop_receiver(self, p_data: int, rec_name: str) -> bool:
        """Deletes a receiver object

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        rec_name : str
            Receiver name

        Returns
        -------
        bool
            True if successful, False otherwise 
        """

        self.pdll.sp_drop_receiver.restype = c_bool
        return self.pdll.sp_drop_receiver( c_void_p(p_data), c_char_p(rec_name.encode()))

    #SPEXPORT int sp_add_heliostat_template(sp_data_t p_data, const char* heliostat_name)
    def add_heliostat_template(self, p_data: int, helio_name: str) -> int:
        """Creates a heliostat template object

        NOTE: CoPylot starts with a default heliostat template at ID = 0, with 'Template 1' as the Heliostat's name. 
        If you add a heliostat template object without dropping this default template, generating a layout will fail 
        because the default heliostat geometry distribution ('solarfield.0.template_rule') is 'Use single template' 
        but the select heliostat geometry ('solarfield.0.temp_which') is not defined.

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        helio_name : str
            heliostat template name

        Returns
        -------
        int
            heliostate template ID 
        """

        self.pdll.sp_add_heliostat_template.restype = c_int
        return self.pdll.sp_add_heliostat_template( c_void_p(p_data), c_char_p(helio_name.encode()))

    #SPEXPORT bool sp_drop_heliostat_template(sp_data_t p_data, const char* heliostat_name)
    def drop_heliostat_template(self, p_data: int, helio_name: str) -> bool:
        """Deletes heliostat template object

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        helio_name : str
            Heliostat template name

        Returns
        -------
        bool
            True if successful, False otherwise 
        """

        self.pdll.sp_drop_heliostat_template.restype = c_bool
        return self.pdll.sp_drop_heliostat_template( c_void_p(p_data), c_char_p(helio_name.encode()))

    #SPEXPORT bool sp_generate_simulation_days(sp_data_t p_data, int *nrecord)
    def generate_simulation_days(self, p_data: int):
        """Report out the days, hours, and weather data used to generate the annual performance estimate
        for the heliostat field layout. 

        ** This function requires that a layout has previously been created **

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance

        Returns
        -------
        list | Each row is a time step; columns are as follows:
            0 | Month (1-12)
            1 | Day of the month (1-N)
            2 | Hour of the day (0-23.999..)
            3 | DNI (W/m^2) direct normal irradiance
            4 | T_db (C)) dry bulb temperature
            5 | V_wind (m/s) wind velocity
            6 | Step_weight (-) relative weight given to each step during layout
            7 | Solar azimuth angle (deg, N=0, +CW)
            8 | Solar zenith angle (deg, 0=zen)
        """
        nrecord = c_int()
        ncol = c_int()
        self.pdll.sp_generate_simulation_days.restype = POINTER(c_number)
        simdays = self.pdll.sp_generate_simulation_days( c_void_p(p_data), byref(nrecord), byref(ncol))
        # self.pdll._sp_free_var.restype = c_void_p
        # self.pdll._sp_free_var( byref(simdays) )
        data = []
        for i in range(nrecord.value):
            data.append(simdays[i*ncol.value:i*ncol.value+ncol.value])

        return data


    #SPEXPORT bool sp_update_geometry(sp_data_t p_data)
    def update_geometry(self, p_data: int) -> bool:
        """Refresh the solar field, receiver, or ambient condition settings based on current parameter settings

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance

        Returns
        -------
        bool
            True if successful, False otherwise 
        """

        self.pdll.sp_update_geometry.restype = c_bool
        return self.pdll.sp_update_geometry( c_void_p(p_data))

    #SPEXPORT bool sp_generate_layout(sp_data_t p_data, int nthreads = 0)
    def generate_layout(self, p_data: int, nthreads: int = 0) -> bool:
        """Create a solar field layout

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        nthreads : int, optional
            Number of threads to use for simulation

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        self.pdll.sp_generate_layout.restype = c_bool
        return self.pdll.sp_generate_layout( c_void_p(p_data), c_int(nthreads))

    #SPEXPORT bool sp_assign_layout(sp_data_t p_data, sp_number_t* pvalues, int nrows, int ncols, int nthreads = 0) //, bool save_detail = true)
    def assign_layout(self, p_data: int, helio_data: list, nthreads: int = 0) -> bool:
        """Run layout with specified positions, (optional canting and aimpoints) 

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        helio_data : list of lists
                heliostat data to assign
                [<template id (int)>, <location X>, <location Y>, <location Z>], <x focal length>, <y focal length>, <cant i>, <cant j>, <cant k>, <aim X>, <aim Y>, <aim Z>]
                NOTE: First 4 columns are required, the rest are optional.
        nthreads : int, optional
            Number of threads to use for simulation

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        nrows = len(helio_data)
        ncols = len(helio_data[0])
        size = nrows*ncols
        arr = (c_number*size)()
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                arr[idx] = c_number(helio_data[r][c])
                idx += 1
        self.pdll.sp_assign_layout.restype = c_bool
        return self.pdll.sp_assign_layout( c_void_p(p_data), pointer(arr), c_int(nrows), c_int(ncols), c_int(nthreads))

    #SPEXPORT sp_number_t* sp_get_layout_info(sp_data_t p_data, int* nhelio, int* ncol, bool get_corners = false)
    def get_layout_info(self, p_data: int, get_corners: bool = False, get_optical_details: bool = False, restype: str = "dataframe"):
        """Get information regarding the heliostat field layout

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        get_corner : bool, optional
            True, output will include heliostat corner infromation, False otherwise
        get_optical_details : bool, optional
            True, output will include focal lengths in X and Y, and will include cant
            panel positions and orientation vectors for each panel. False, no info.
                Format:
                focal x, focal y, panel1.x, .y, .z, panel1.i, .j, .k, panel2.x, ....
        restype : str, optional 
            result format type, supported options: "matrix", "dictionary", "dataframe"

        Returns
        -------
        pandas.DataFrame()
            heliostat field layout infromation in dataframe format
        dict
            heliostat field layout infromation in dictionary format
        list of list (matrix), list of strings
            heliostat field layout infromation with data in a matrix and column names in a header list of strings
        """

        nrows = c_int()
        ncols = c_int()
        # Get data
        self.pdll.sp_get_layout_info.restype = POINTER(c_number)
        parr = self.pdll.sp_get_layout_info( c_void_p(p_data), byref(nrows), byref(ncols),  c_bool(get_corners), c_bool(get_optical_details))
        # Get header
        self.pdll.sp_get_layout_header.restype = c_char_p
        header = self.pdll.sp_get_layout_header( c_void_p(p_data), c_bool(get_corners), c_bool(get_optical_details)).decode()
        header = header.split(',')
        if restype.lower().startswith("mat"):
            # output matrix
            mat = []
            for r in range(nrows.value):
                row = []
                for c in range(ncols.value):
                    row.append( float(parr[ncols.value * r + c]))
                mat.append(row)
            return mat, header
        elif restype.lower().startswith("dic") or restype.lower().startswith("dat"):
            # output dictionary
            ret = {}
            for c,key in enumerate(header):
                ret[key] = []
                for r in range(nrows.value):
                    ret[key].append( float(parr[ncols.value * r + c]))
            if restype.lower().startswith("dic"):
                return ret
            else:
                df = pd.DataFrame(ret)
                df.set_index(header[0])
                return df

    #SPEXPORT bool sp_simulate(sp_data_t p_data, int nthreads = 1, bool update_aimpoints = true)
    def simulate(self, p_data: int, nthreads: int = 1, update_aimpoints: bool = True) -> bool:
        """Calculate heliostat field performance

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        nthreads : int, optional
            number of threads to use for simulation
        update_aimpoints : bool, optional
            True, aimpoints update during simulation, False otherwise 

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        self.pdll.sp_simulate.restype = c_bool
        return self.pdll.sp_simulate( c_void_p(p_data), c_int(nthreads), c_bool(update_aimpoints))

    #SPEXPORT const char *sp_summary_results(sp_data_t p_data)
    def summary_results(self, p_data: int, save_dict: bool = True):
        """Prints table of summary results from each simulation

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        save_dict : bool, optional
            True, return results as dictionary

        Returns
        -------
        dict
            dictionary containing simulation summary results (default)
        None
            prints summary table to terminal (save_dict = False)
        """

        self.pdll.sp_summary_results.restype = c_char_p
        ret = self.pdll.sp_summary_results( c_void_p(p_data)).decode()
        # save result table to dictionary
        if save_dict:
            items = ret.split('\n')
            res_dict = {}
            for i in items:
                key_val = i.split(',')
                try:
                    res_dict[key_val[0]] = float(key_val[1])
                except:
                    res_dict[key_val[0]] = key_val[1]
            return res_dict
        else:    # print results table
            return print(ret)

    #SPEXPORT sp_number_t* sp_detail_results(sp_data_t p_data, int* nrows, int* ncols, sp_number_t* selhel = NULL, int nselhel = 0)
    def detail_results(self, p_data: int, selhel: list = None, restype: str = "dataframe", get_corners: bool = False):
        """Get heliostat field layout detail results

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        selhel : list, optional
            Selected heliostat ID
        restype : str, optional
            result format type, supported options "matrix", "dictionary", "dataframe" \n
        get_corner : bool, optional
            True, output will include heliostat corner infromation, False otherwise

        Returns
        -------
        pandas.DataFrame()
            detailed heliostat field results in dataframe format
        dict
            detailed heliostat field results in dictionary format
        list of list (matrix), list of strings
            detailed heliostat field results with data in a matrix and column names in a header list of strings
        """

        # handling selected heliostats
        if selhel == None:
            nselhel = 0
            arr = c_number(0)
        else:
            nselhel = len(selhel)
            arr = (c_number*nselhel)()
            arr[:] = selhel # set all at once

        nrows = c_int()
        ncols = c_int()
        # Get data
        self.pdll.sp_detail_results.restype = POINTER(c_number)
        res_arr = self.pdll.sp_detail_results( c_void_p(p_data), byref(nrows), byref(ncols), pointer(arr), c_int(nselhel), c_bool(get_corners))
        try:
            # Get header
            self.pdll.sp_detail_results_header.restype = c_char_p
            header = self.pdll.sp_detail_results_header( c_void_p(p_data), c_bool(get_corners)).decode()
            header = header.split(',')
            if restype.lower().startswith("mat"):
                # output matrix
                mat = []
                for r in range(nrows.value):
                    row = []
                    for c in range(ncols.value):
                        row.append( float(res_arr[ncols.value * r + c]))
                    mat.append(row)
                return mat, header
            elif restype.lower().startswith("dic") or restype.lower().startswith("dat"):
                # output dictionary
                ret = {}
                for c,key in enumerate(header):
                    ret[key] = []
                    for r in range(nrows.value):
                        ret[key].append( float(res_arr[ncols.value * r + c]))
                if restype.lower().startswith("dic"):
                    return ret
                else:
                    df = pd.DataFrame(ret)
                    df.set_index(header[0])
                    return df
        except:
            print("detail_results API called failed to return correct information.")

    #SPEXPORT sp_number_t* sp_get_fluxmap(sp_data_t p_data, int* nrows, int* ncols, int rec_id = 0)
    def get_fluxmap(self, p_data: int, rec_id: int = 0) -> list:
        """Retrieve the receiver fluxmap, optionally specifying the receiver ID to retrieve

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        rec_id : int, optional
            receiver ID to retrieve

        Returns
        -------
        list of lists (matrix)
            receiver fluxmap
        """

        nrows = c_int()
        ncols = c_int()
        self.pdll.sp_get_fluxmap.restype = POINTER(c_number)
        res = self.pdll.sp_get_fluxmap( c_void_p(p_data), byref(nrows), byref(ncols), c_int(rec_id))
        # output matrix
        mat = []
        for r in range(nrows.value):
            row = []
            for c in range(ncols.value):
                row.append( float(res[ncols.value * r + c]))
            mat.append(row)
        return mat
    
    #SPEXPORT void sp_clear_land(sp_data_t p_data, const char* type = NULL)
    def clear_land(self, p_data: int, clear_type: str = None) -> None:
        """Reset the land boundary polygons, clearing any data

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        clear_type : str, optional
            specify land boundaries to clear, options are 'None' (default), 'inclusion', or 'exclusion'

        Returns
        -------
        None
        """

        self.pdll.sp_clear_land( c_void_p(p_data), c_char_p(clear_type.encode()))

    #SPEXPORT bool sp_add_land(sp_data_t p_data, const char* type, sp_number_t* polygon_points, int* npts , int* ndim, bool is_append = true)
    def add_land(self, p_data: int, add_type: str, poly_points: list, is_append: bool = True) -> bool:
        """Add land inclusion or a land exclusion region within a specified polygon

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        add_type : str
            specify type of added land boundary, options are inclusion' or 'exclusion'
        poly_points : list of lists (matrix)
            list of polygon points [[x1,y1],[x2,y2],...] or [[x1,y1,z1],[x2,y2,z2],...]
        is_append : bool, optional
            Append (True) or overwrite (False) the existing regions

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        npts = len(poly_points)
        ndim = len(poly_points[0])
        size = npts*ndim
        PParr = (c_number*size)()
        idx = 0
        for r in range(npts):
            for c in range(ndim):
                PParr[idx] = c_number(poly_points[r][c])
                idx += 1
        
        self.pdll.sp_add_land.restype = c_bool
        return self.pdll.sp_add_land( c_void_p(p_data), c_char_p(add_type.encode()), pointer(PParr), byref(c_int(npts)), byref(c_int(ndim)), c_bool(is_append))

    #SPEXPORT sp_number_t* sp_heliostats_by_region(sp_data_t p_data, const char* coor_sys, int* lenret,
    #                                                sp_number_t* arguments = NULL, int* len_arg = NULL,
    #                                                const char* svgfname_data = NULL, sp_number_t* svg_opt_tab = NULL);
    def heliostats_by_region(self, p_data: int, coor_sys: str = 'all', **kwargs):
        """Returns heliostats that exists within a region

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        coor_sys : str, optional
            Options are
                'all' (no additional infromation required),
                'cylindrical' (provide 'arguments' [rmin, rmax, azmin radians, azmax radians]),
                'cartesian' (provide 'arguments' [xmin, xmax, ymin, ymax[, zmin, zmax]]),
                'polygon' (provide 'arguments' [[x1,y1],[x2,y2],...]),
                'svg' (provide 'svgfname_data' string with 'scale-x scale-y;offset-x offset-y;<svg path 1>;<svg path 2>;...',
                'svgfile' (provide 'svgfname_data' string of filename with path, optional 'svg_opt_tab' table (list) [scale-x, scale-y, offset-x, offset-y])

        kwargs
        ------
        arguments : list
            depends on coor_sys selected
        svgfname_data : str
            either svg data or a svg file path
        svg_opt_tab : list, optional
            svg optional table for scale and offset [x-scale, y-scale, x-offset, y-offset]
        restype : str, optional 
            result format type, supported options: "matrix", "dictionary", "dataframe"

        Returns
        -------
        pandas.DataFrame()
            heliostat field layout infromation in dataframe format (default)
        dict
            heliostat field layout infromation in dictionary format
        list of list (matrix), list of strings
            heliostat field layout infromation with data in a matrix and column names in a header list of strings
        """

        argsdict = {
            'arguments': [],
            'svgfname_data': '', 
            'svg_opt_tab' : [],
            'restype': 'dataframe'
        }
        argsdict.update(kwargs)

        # flatten polygon points
        if coor_sys == 'polygon':
            temp = []
            for r in range(len(argsdict['arguments'])):
                for c in range(len(argsdict['arguments'][0])):
                    temp.append(argsdict['arguments'][r][c])
            argsdict['arguments'] = temp
        
        len_arg = len(argsdict['arguments'])
        arg_arr = (c_number*len_arg)()
        arg_arr[:] = argsdict['arguments']

        lenret = c_int()
        self.pdll.sp_heliostats_by_region.restype = POINTER(c_number)
        if coor_sys == 'all':
            res = self.pdll.sp_heliostats_by_region( c_void_p(p_data), c_char_p(coor_sys.encode()), byref(lenret) )
        elif coor_sys == 'cylindrical' or coor_sys == 'cartesian' or coor_sys == 'polygon':
            res = self.pdll.sp_heliostats_by_region( c_void_p(p_data), c_char_p(coor_sys.encode()), byref(lenret), pointer(arg_arr), byref(c_int(len_arg)))
        elif coor_sys == 'svg' or coor_sys == 'svgfile':
            if len(argsdict['svg_opt_tab'] == 0):
                res = self.pdll.sp_heliostats_by_region( c_void_p(p_data), c_char_p(coor_sys.encode()), byref(lenret), pointer(arg_arr), byref(c_int(len_arg)), c_char_p(argsdict['svgfname_data'].encode()) )
            elif len(argsdict['svg_opt_tab']) == 4:
                svg_tab = (c_number*4)()
                svg_tab[:] = argsdict['svg_opt_tab']
                res = self.pdll.sp_heliostats_by_region( c_void_p(p_data), c_char_p(coor_sys.encode()), byref(lenret), pointer(arg_arr), byref(c_int(len_arg)), c_char_p(argsdict['svgfname_data'].encode()), pointer(svg_tab) )
            else:
                print('svg_opt_tab must have the following form: [scale-x, scale-y, offset-x, offset-y]')
        else:
            print('Invalid region type specified. Expecting one of [cylindrical, cartesian, polygon, svg, svgfile]')

        # Unpacking results - id, location x, location y, location z
        header = ['id','location-x','location-y','location-z']
        if argsdict['restype'].lower().startswith("mat"):
            # output matrix
            mat = []
            for r in range(int(lenret.value/4)):
                row = []
                for c in range(4):
                    row.append( float(res[4 * r + c]))
                mat.append(row)
            return mat, header
        elif argsdict['restype'].lower().startswith("dic") or argsdict['restype'].lower().startswith("dat"):
            # output dictionary
            ret = {}
            for c,key in enumerate(header):
                ret[key] = []
                for r in range(int(lenret.value/4)):
                    ret[key].append( float(res[4 * r + c]))
            if argsdict['restype'].lower().startswith("dic"):
                return ret
            else:
                df = pd.DataFrame(ret)
                df.set_index(header[0])
                return df

    #SPEXPORT bool sp_modify_heliostats(sp_data_t p_data, sp_number_t* helio_data, int* nhel, int* ncols, const char* table_hdr)
    def modify_heliostats(self, p_data: int, helio_dict: dict) -> bool:
        """Modify attributes of a subset of heliostats in the current layout

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        helio_dict : dict
            Heliostat modified attributes, dictionary keys are as follows:
                    'id',               - Required
                    'location-x',
                    'location-y',
                    'location-z',
                    'aimpoint-x',
                    'aimpoint-y',
                    'aimpoint-z',
                    'soiling',
                    'reflectivity',
                    'enabled'

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        ncols = len(helio_dict.keys())
        nhel = len(helio_dict[next(iter(helio_dict))])
        table_hdr = ""
        for key in helio_dict.keys():
            table_hdr += key
            if (key != list(helio_dict)[-1]):
                table_hdr += ','
        
        size = ncols*nhel
        helio_data = (c_number*size)()
        idx = 0
        for h in range(nhel):
            for key in helio_dict.keys():
                helio_data[idx] = c_number(helio_dict[key][h])
                idx += 1

        self.pdll.sp_modify_heliostats.restype = c_bool
        return self.pdll.sp_modify_heliostats( c_void_p(p_data), pointer(helio_data), byref(c_int(nhel)), byref(c_int(ncols)), c_char_p(table_hdr.encode()) )

    #SPEXPORT bool sp_calculate_optical_efficiency_table(sp_data_t p_data, int ud_n_az, int ud_n_zen);
    def calculate_optical_efficiency_table(self, p_data: int,  ud_n_az: int = 0, ud_n_zen: int = 0) -> bool:
        """Calculates optical efficiency table  based an even spacing of azimuths and zeniths (elevation) angles.

         Parameters
         ----------
         p_data : int
             memory address of SolarPILOT instance
         ud_n_az : int
             user-defined number of azimuth sampling points
         ud_n_zen : int
             user-defined number of zenith (elevation) sampling points

         Returns
         -------
         bool
             True if successful, False otherwise
         """

        self.pdll.sp_calculate_optical_efficiency_table.restype = c_bool
        if ud_n_az == 0 and ud_n_zen == 0:
            return self.pdll.sp_calculate_optical_efficiency_table( c_void_p(p_data))
        else:
            return self.pdll.sp_calculate_optical_efficiency_table( c_void_p(p_data), c_int(ud_n_az), c_int(ud_n_zen))

    #SPEXPORT sp_number_t* sp_get_optical_efficiency_table(sp_data_t p_data, int* nrows, int* ncols)
    def get_optical_efficiency_table(self, p_data: int) -> dict:
        """Retrieve the field optical efficiency table as a function of azimuth and elevation angles

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        
        Returns
        -------
        dictionary with the following keys:

            #. ``azimuth``: list, Solar azimuth angle [deg]
            #. ``elevation``: list, Solar elevation angle [deg]
            #. ``eff_data``: list of lists, Solar field optical efficiency at a specific azimuth (rows) and elevation (cols) angles [-]
        """

        nrows = c_int()
        ncols = c_int()
        self.pdll.sp_get_optical_efficiency_table.restype = POINTER(c_number)
        res = self.pdll.sp_get_optical_efficiency_table( c_void_p(p_data), byref(nrows), byref(ncols))
        # Formatting output
        elevation = []
        azimuth = []
        eff_data = []
        for r in range(nrows.value):
            row = []
            for c in range(ncols.value):
                if r == 0 and c == 0:
                    pass
                elif r == 0:
                    elevation.append( float(res[ncols.value * r + c]))
                elif r != 0 and c == 0:
                    azimuth.append( float(res[ncols.value * r + c]))
                else:
                    row.append( float(res[ncols.value * r + c]))
            if r != 0:
                eff_data.append(row)
        
        return {'azimuth': azimuth, 'elevation': elevation, 'eff_data': eff_data}

    #SPEXPORT void sp_calculate_get_optical_efficiency_table(sp_data_t p_data, const size_t n_azi, const size_t n_elev, double* azimuths, double* elevation, double* eff_matrix)
    def calculate_get_optical_efficiency_table(self, p_data: int, n_azi: int, n_elev: int) -> dict:
        """Calculates and retrieves the field optical efficiency table as a function of azimuth and elevation angles

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        n_azi : int
            Number of azimuth angles used to create efficiency table (evenly spaced)
        n_elev : int
            Number of elevation angles used to create efficiency table (evenly spaced)
        
        Returns
        -------
        dictionary with the following keys:

            #. ``azimuth``: list, Solar azimuth angle [deg]
            #. ``elevation``: list, Solar elevation angle [deg]
            #. ``eff_data``: list of lists, Solar field optical efficiency at a specific azimuth (rows) and elevation (cols) angles [-]
        """
        # self.calculate_optical_efficiency_table(p_data, n_azi, n_elev)
        # return self.get_optical_efficiency_table(p_data)
        azi = (c_number * n_azi)()
        elev = (c_number * n_elev)()
        eff_mat = (c_number * n_elev * n_azi)()
        self.pdll.sp_calculate_get_optical_efficiency_table( c_void_p(p_data), c_int(n_azi), c_int(n_elev), pointer(azi), pointer(elev), pointer(eff_mat))
        azimuth = azi[0:n_azi]
        elevation = elev[0:n_elev]
        eff_matrix = []
        for r in range(n_azi):
            eff_matrix.append(eff_mat[r][0:n_elev])

        return {'azimuth': azimuth, 'elevation': elevation, 'eff_data': eff_matrix}


    #SPEXPORT bool sp_save_optical_efficiency_table(sp_data_t p_data, const char* sp_fname, const char* table_name)
    def save_optical_efficiency_table(self, p_data: int, sp_fname: str, modelica_table_name: str = 'none') -> bool:
        """
        Saves optical efficiency table as a CSV file in the following format:
            First row: Elevation angles (with leading zero)
            First column: Azimuth angles
            Rest of columns: Optical efficiency corresponding to elevation angle       

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        sp_fname : str
            filename to save efficiency table
        modelica_table_name : str (optional)
            Modelica table name for table output file consistent with Modelica formatting requirements. 
            If not provided, then table format follows get_optical_efficiency_table(). 
            Otherwise, table format contains extra header lines and elevation angle to be in ascending order (required by Modelica).

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        self.pdll.sp_save_optical_efficiency_table.restype = c_bool
        return self.pdll.sp_save_optical_efficiency_table( c_void_p(p_data), c_char_p(sp_fname.encode()), c_char_p(modelica_table_name.encode()))

    #SPEXPORT bool sp_save_from_script(sp_data_t p_data, const char* sp_fname)
    def save_from_script(self, p_data: int, sp_fname: str) -> bool:
        """Save the current case as a SolarPILOT .spt file

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        sp_fname : str
            filename to save SolarPILOT case

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        self.pdll.sp_save_from_script.restype = c_bool
        return self.pdll.sp_save_from_script( c_void_p(p_data), c_char_p(sp_fname.encode()))

    #SPEXPORT bool sp_load_from_script(sp_data_t p_data, const char* sp_fname)
    def load_from_script(self, p_data: int,  sp_fname: str) -> bool:
        """Load a SolarPILOT .spt file

         Parameters
         ----------
         p_data : int
             memory address of SolarPILOT instance
         sp_fname : str
             filename to load SolarPILOT case

         Returns
         -------
         bool
             True if successful, False otherwise
         """

        self.pdll.sp_load_from_script.restype = c_bool
        return self.pdll.sp_load_from_script( c_void_p(p_data), c_char_p(sp_fname.encode()))

    #SPEXPORT bool sp_dump_varmap(sp_data_t p_data, const char* sp_fname)
    def dump_varmap_tofile(self, p_data: int, fname: str) -> bool:
        """Dump the variable structure to a text csv file

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance
        fname : str
            filename to save variables

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        self.pdll.sp_dump_varmap.restype = c_bool
        return self.pdll.sp_dump_varmap( c_void_p(p_data), c_char_p(fname.encode()))
    
    # SPEXPORT bool sp_export_soltrace(sp_data_t p_data, const char* sp_fname)
    def export_soltrace(self, p_data: int, fname: str) -> bool:
        """
        """
        self.pdll.sp_export_soltrace.restype = c_bool
        return self.pdll.sp_export_soltrace( c_void_p(p_data), c_char_p(fname.encode()))

    # SPEXPORT bool sp_load_soltrace_context(sp_data_t p_data, st_context_t* solt_cxt)
    def load_soltrace_context(self, p_data: int, solt_cxt : int) -> bool:

        self.pdll.sp_load_soltrace_context.restype = c_bool
        return self.pdll.sp_load_soltrace_context( c_void_p(p_data), c_void_p(solt_cxt))

    def load_soltrace_structure(self, p_data: int):
        """
        **Follows CreateSTSystem in the cpp code / STObject.cpp**

        Before calling this method, p_data must (1) contain a valid solar field layout and (2) a performance simulation
        must have been run at the desired sun position and irradiation condition. This method will pull information from 
        the most recent analytical performance simulation, including: 
            - list of heliostats included in the layout
            - current receiver aim point for each heliostat
            - attenuation and soiling efficiency for each heliostat
            - current sun position

        This method takes a SolarPILOT layout and generates an instance of the PySoltrace API object needed to directly 
        run SolTrace simulations from the SolTrace API. This differs from the function CreateSTSystem implemented in 
        SolarPILOT in that this method creates a PySolTrace object while SolarPILOT creates a 'coretrace' context via
        the c++ API. 

        Once created, the PySolTrace object can be manipulated to add or change geometry (e.g., adding receiver 
        complexity), to change optical property settings, or to change anything related to the SolTrace configuration
        before a raytrace simulation is run. This method serves as a bridge between SolarPILOT and SolTrace python 
        functionality.

        Parameters
        ----------
        p_data : int
            memory address of SolarPILOT instance

        Returns
        -------
        pysoltrace.PySolTrace 
            Returns an instance of the PySolTrace class that is populated with heliostats, optical properties, 
            receiver geometry, and sun properties that mirror the SolarPILOT solar field. 

        """

        P = pysoltrace.PySolTrace()

        sun_type = int(self.data_get_number(p_data, "ambient.0.sun_type"))
        
        P.add_sun()
        # sun_type; {0=point sun, 1=limb darkened sun, 2=square wave sun, 3=user sun}
        sigma = self.data_get_number(p_data, "ambient.0.sun_rad_limit")
        shape = 'i'	#invalid
        if sun_type == 2:
            shape = 'p'  #Pillbox sun
        elif sun_type == 0:
            shape = 'g' #Point sun -- doesn't matter just use something here. it is disabled later.
        elif sun_type == 4: 
            shape = 'g' #Gaussian sun
        elif sun_type == 1:
        	#Limb-darkened sun
            #Create a table based on the limb-darkened profile and set as a user sun

            shape = 'd'
            np = 26
            R = 4.65e-3     #maximum subtended angle
            dr = R/(np-1)
            angle = [0. for i in range(np)]
            intens = [0. for i in range(np)]
            for i in range(np): 
                angle[i] = dr*i
                intens[i] = 1.0 - 0.5138*pow(angle[i]/R, 4)
                angle[i] *= 1000.;	#mrad
            
            intens[np-1] = 0.
            
            #Fill into the sun object
            P.sun.user_intensity_table.clear()
            # Sun.SunShapeIntensity.resize(np);
            # Sun.SunShapeAngle.resize(np);
            for i in range(np):
                P.sun.user_intensity_table.append([angle[i],intens[i]])
        elif sun_type == 5:
            #Buie sun
            shape = 'd'
            
            #Fill into the sun object
            P.sun.user_intensity_table.clear()
            dt_s = .2
            dt_tr = .05
            dt_cs = 1.
            angle_max = 43.6
            delta_theta_tr = 1.

            theta = -dt_s  #set so first adjustment is back to 0

            #correct for chi based on tonatiuh polys
            csr = self.data_get_number(p_data, "ambient.0.sun_csr")
            if (csr > 0.145):
                chi = -0.04419909985804843 + csr * (1.401323894233574 + csr * (-0.3639746714505299 + csr * (-0.9579768560161194 + 1.1550475450828657 * csr)))
            elif (csr > 0.035):
                chi = 0.022652077593662934 + csr * (0.5252380349996234 + (2.5484334534423887 - 0.8763755326550412 * csr) * csr)
            else:
                chi = 0.004733749294807862 + csr * (4.716738065192151 + csr * (-463.506669149804 + csr * (24745.88727411664 + csr * (-606122.7511711778 + 5521693.445014727 * csr))))

            _buie_kappa = 0.9*math.log(13.5 * chi)*pow(chi, -0.3)
            _buie_gamma = 2.2*math.log(0.52 * chi)*pow(chi, 0.43) - 0.1

            while(theta < angle_max):

                if (theta < 4.65 - delta_theta_tr / 2.):
                    theta += dt_s
                
                elif (theta > 4.65 + delta_theta_tr / 2.):
                    theta += dt_cs
                    dt_cs *= 1.2   #take larger steps as we get away from the transition region
                else:
                    theta += dt_tr

                if (theta > 4.65):
                    theta = angle_max + .000001 if (theta > angle_max) else theta
                    #in the circumsolar region
                    P.sun.user_intensity_table.append([theta, math.exp(_buie_kappa)*pow(theta, _buie_gamma)])
                else:
                    #in the solar disc
                    P.sun.user_intensity_table.append([theta, math.cos(0.326 * theta) / math.cos(0.308 * theta)])

        elif sun_type == 3:
            #User sun
            shape = 'd'
            pass  #user must have filled user_intensity_table manually
        else:
            return False
        
        #set other sun parameters
        P.sun.shape = shape
        P.sun.sigma = sigma
        
        #--- Set the sun position ---
        solaz = self.data_get_number(p_data, "fluxsim.0.flux_solar_az")
        solel = self.data_get_number(p_data, "fluxsim.0.flux_solar_el")
        sun = pysoltrace.Point()
        sun.z = math.cos(math.radians(90.-solel))
        sun.x = math.sin(math.radians(solaz))*math.sqrt(1-sun.z**2)
        sun.y = math.cos(math.radians(solaz))*math.sqrt(1-sun.z**2)

        sun.unitize(inplace=True)
        
        P.sun.position.x = sun.x*1.e4
        P.sun.position.y = sun.y*1.e4
        P.sun.position.z = sun.z*1.e4


        # --- Set up optical property set ---
        
        # The optical property set describes the behavior of a surface (front and back sides) optically.
        # Reflective properties, transmissivity, slope error and specularity, and error type are specified. 
        # Several irrelevant properties must also be set, including refraction and grating properties.
        
        # Create an optical property set for each heliostat template

        # helios = self.heliostats_by_region(p_data)
        
        # merge info on the heliostat field from the detailed results (aim points) and layout_info (optical characteristics)
        h1 = self.detail_results(p_data)
        # 'id', 'x_location', 'y_location', 'z_location', 'x_aimpoint',
        # 'y_aimpoint', 'z_aimpoint', 'i_tracking_vector', 'j_tracking_vector',
        # 'k_tracking_vector', 'layout_metric', 'power_to_receiver',
        # 'power_reflected', 'energy', 'efficiency_annual', 'efficiency',
        # 'cosine', 'intercept', 'reflectance', 'attenuation', 'blocking',
        # 'shading', 'clouds'
        h1 = h1.set_index('id')
        h2 = self.get_layout_info(p_data, get_optical_details=True, restype="dataframe")
        h2 = h2.set_index('id')
        helios = h1.combine_first(h2)
        
        nhtemp = len(helios)
        optics_map = {}
        for i in range(nhtemp):
            oname = f"heliostat_{i:d}"
            P.add_optic(oname)
            #map the optics pointer to the heliostat template name
            optics_map[oname] = P.optics[i]

        for ii in range(nhtemp):
            H = helios.iloc[ii]
            
            """
            The optical error in SolTrace is described in spherical coordinates, so the total error 
            budget should represent the weighted average of the X and Y components. To average, divide
            the final terms by sqrt(2). If both X and Y errors are the same, then the result will be
            sigma_spherical = sigma_x = sigma_y. Otherwise, the value will fall between sigma_x and 
            sigma_y.
            """
            
            refl = H.reflectance  #reflectivity * soiling
            # scale reflectance by attenuation for this heliostat
            refl *= H.attenuation 
            #Note that the reflected energy is also reduced by the fraction of inactive heliostat aperture. Since
            #we input the actual heliostat dimensions into soltrace, apply this derate on the reflectivity.
            refl *= self.data_get_number(p_data, "heliostat.0.reflect_ratio") 
            
            errang = [
                self.data_get_number(p_data, "heliostat.0.err_azimuth"),
                self.data_get_number(p_data, "heliostat.0.err_elevation"),
            ]
            errsurf = [
                self.data_get_number(p_data, "heliostat.0.err_surface_x"),
                self.data_get_number(p_data, "heliostat.0.err_surface_y"),
            ]
            errrefl = [
                self.data_get_number(p_data, "heliostat.0.err_reflect_x"),
                self.data_get_number(p_data, "heliostat.0.err_reflect_y"),
            ]

            errnorm = math.sqrt( errang[0]*errang[0] + errang[1]*errang[1]  + errsurf[0]*errsurf[0] + errsurf[1]*errsurf[1] )*1000. #mrad  normal vector error
            errsurface = math.sqrt( errrefl[0]*errrefl[0] + errrefl[1]*errrefl[1] ) * 1000.  #mrad - reflected vector error (specularity)
            
            """
            The Hermite model definitions treat x and y components as conical error, such that the following definition holds:

            sigma_tot^2 = ( sigma_x^2 + sigma_y^2 )/2

            This definition is unconventional, but a conversion factor of 1/sqrt(2) is required when expressing x and y component
            errors from the Hermite model in total error for SolTrace.
            """

            errnorm *= 1./math.sqrt(2)
            errsurface *= 1./math.sqrt(2)

            #Add the front
            st_err_type = self.data_get_number(p_data, "heliostat.0.st_err_type")   #Gaussian=0;Pillbox=1
            if st_err_type == 0:  #gaussian
                P.optics[ii].front.dist_type = 'g'
            elif st_err_type == 1: #pillbox
                P.optics[ii].front.dist_type = 'p'
            P.optics[ii].front.transmissivity = 0.
            P.optics[ii].front.reflectivity = refl
            P.optics[ii].front.slope_error = errnorm
            P.optics[ii].front.spec_error = errsurface

            P.optics[ii].back.dist_type = 'g'
            P.optics[ii].back.reflectivity = 0.
            P.optics[ii].back.transmissivity = 0.
            P.optics[ii].back.slope_error = 100.
            P.optics[ii].back.spec_error = 0.

            
        # --- Set the heliostat stage ---
        # this contains all heliostats regardless of differeing geometry or optical properties

        h_stage = P.add_stage()
        #global origin
        h_stage.position.x = 0.
        h_stage.position.y = 0.
        h_stage.position.z = 0.
        #global aim, leave as 0,0,1
        h_stage.aim.x = 0.
        h_stage.aim.y = 0.
        h_stage.aim.z = 1.
        #no z rotation
        h_stage.zrot = 0.
        #{virtual stage, multiple hits per ray, trace through} UI checkboxes
        h_stage.is_virtual = False
        h_stage.is_multihit = True 
        h_stage.is_tracethrough = False
        #name
        h_stage.name = "Heliostat field"

        # 	--- Add elements to the stage ---
        nh = len(helios)
        
        # determine whether each heliostat contains additional facet details
        isdetail = bool(self.data_get_number(p_data, "heliostat.0.is_faceted"))
            
        ncantx = int(self.data_get_number(p_data, "heliostat.0.n_cant_x")) if isdetail else 1
        ncanty = int(self.data_get_number(p_data, "heliostat.0.n_cant_y")) if isdetail else 1
        
        for i in range(nh):
            H = helios.iloc[i]


            #Get values that apply to the whole heliostat
            enabled = True
            
            #compute tracking vector from aim point
            V = pysoltrace.Point()
            V.x = H.x_aimpoint - H.x_location
            V.y = H.y_aimpoint - H.y_location
            V.z = H.z_aimpoint - H.z_location
            V.unitize(inplace=True)
            V.x += sun.x
            V.y += sun.y
            V.z += sun.z 
            V = V/2.
            V.unitize(inplace=True)
            
            zrot = P.util_calc_zrot_azel(V)  #degrees

            shape = 'c' if bool(self.data_get_number(p_data, "heliostat.0.is_round")) else 'r'

            opticname = f"heliostat_{i:d}"
            track_zen = math.acos(V.z)
            track_az = math.atan2(V.x,V.y)

            for j in range(ncantx):
                for k in range(ncanty):

                    element = h_stage.add_element()

                    element.enabled = enabled
                    
                    if isdetail:
                        #Calculate unique positions and aim vectors for each facet
                        panel_name = f"panel_{k:d}_{j:d}"
                        Floc = pysoltrace.Point( H[f"{panel_name}_x"], H[f"{panel_name}_y"], H[f"{panel_name}_z"] )
                        Faim = pysoltrace.Point( H[f"{panel_name}_i"], H[f"{panel_name}_j"], H[f"{panel_name}_k"] )
                        Faim = self.__unitvect(Faim) 

                        #Rotate to match heliostat rotation
                        Floc = P.util_rotation_arbitrary(track_zen, pysoltrace.Point(1,0,0), pysoltrace.Point(), Floc)
                        Faim = P.util_rotation_arbitrary(track_zen, pysoltrace.Point(1,0,0), pysoltrace.Point(), Faim)
                        Floc = P.util_rotation_arbitrary(math.pi - track_az, pysoltrace.Point(0,0,1), pysoltrace.Point(), Floc)
                        Faim = P.util_rotation_arbitrary(math.pi - track_az, pysoltrace.Point(0,0,1), pysoltrace.Point(), Faim)
                        
                        element.position.x = H.x_location + Floc.x
                        element.position.y = H.y_location + Floc.y
                        element.position.z = H.z_location + Floc.z

                        element.aim.x = element.position.x + Faim.x*1000.
                        element.aim.y = element.position.y + Faim.y*1000.
                        element.aim.z = element.position.z + Faim.z*1000.
                    else:
                        element.position.x = H.x_location
                        element.position.y = H.y_location
                        element.position.z = H.z_location
                    
                        element.aim.x = H.x_location + V.x*1000.
                        element.aim.y = H.y_location + V.y*1000.
                        element.aim.z = H.z_location + V.z*1000.

                    element.zrot = zrot
            
                    element.aperture = shape
                    
                    #Set up the surface description
                    if self.data_get_number(p_data, "heliostat.0.is_round"):
                        element.aperture_params[0] = self.data_get_number(p_data, "heliostat.0.width")
                    else:
                        if isdetail:
                            #Image size is for each individual facet.
                            element.aperture_params[0] = H.panel_width 
                            element.aperture_params[1] = H.panel_height
                        else:
                            element.aperture_params[0] = self.data_get_number(p_data, "heliostat.0.width")
                            element.aperture_params[1] = self.data_get_number(p_data, "heliostat.0.height")
            
                    #Model surface as either flat or parabolic focus in X and/or Y
                    #double spar[] ={0., 0., 0., 0., 0., 0., 0., 0.};
                    # Flat=0;At slant=1;Group average=2;User-defined=3

                    # modify the get_layout_info method to give back focal lengths and possibly canting information?

                    if self.data_get_number(p_data, "heliostat.0.focus_method") == 0:
                    	#Flat
                        element.surface = 'f'
                    else:	
                        #Not flat
                        #coefs are 1/2*f where f is focal length in x or y
                        element.surface_params[0] = 0.5/H.focal_x
                        element.surface_params[1] = 0.5/H.focal_y
                        element.surface = 'p'
                    
                    element.interaction = 2;	#1 = refract, 2 = reflect
                    element.optic = optics_map[ opticname ]

                
        #--- Set the receiever stages ---
        r_stage = P.add_stage()
        #Global origin
        r_stage.position.x = 0.
        r_stage.position.y = 0.
        r_stage.position.z = 0.
        #Aim point
        r_stage.aim.x = 0.
        r_stage.aim.y = 0.
        r_stage.aim.z = 1.
        #No z rotation
        r_stage.zrot = 0.
        #{virtual stage, multiple hits per ray, trace through} UI checkboxes
        r_stage.is_virtual = False 
        r_stage.is_multihit = True 
        r_stage.is_tracethrough = False
        #Name
        r_stage.name = "Receiver"

        #only the first receiver is considered
        element = r_stage.add_element()

        #Get the receiver

        #Get the receiver geometry type
        rectype = self.data_get_number(p_data, "receiver.0.rec_type") 
        if rectype == 0:
            recgeom = 0  #CYLINDRICAL_CLOSED
        elif rectype == 1:
            recgeom = 2  #CYLINDRICAL_CAV
        elif rectype == 2:
            recgeom = 3  #PLANE_RECT
        

        #append an optics set, required for the receiver
        recname = self.data_get_string(p_data, "receiver.0.class_name")
        copt = P.add_optic(recname)

        if recgeom == 0:  #CYLINDRICAL_CLOSED:
            #Add optics stage
            
            #set the optical properties. This should be a diffuse surface, make it a pillbox distribution w/ equal angular reflection probability.
            copt.front.dist_type = 'g'
            copt.front.reflectivity = 1.-self.data_get_number(p_data, "receiver.0.absorptance")
            copt.front.slope_error = 100.
            copt.front.spec_error = 100.
            #back
            copt.back.dist_type = 'g'
            copt.back.reflectivity = 1.-self.data_get_number(p_data, "receiver.0.absorptance")
            copt.back.slope_error = 100.
            copt.back.spec_error = 100.

            #displace by radius, inside is front, x1 and x2 = 0 for closed cylinder ONLY
            #Add a closed cylindrical receiver to the stage 
            diam = self.data_get_number(p_data, "receiver.0.rec_diameter")

            element.enabled = True
            element.position.x = self.data_get_number(p_data, "receiver.0.rec_offset_x_global")
            element.position.y = self.data_get_number(p_data, "receiver.0.rec_offset_y_global") - diam/2.
            element.position.z = self.data_get_number(p_data, "receiver.0.optical_height") #optical height includes z offset
            
            #calculate the aim point. we need to rotate the receiver from a horizontal position into a vertical
            #position. The aim vector defines the Z axis with respect to the SolTrace receiver coordinates, and
            #in SolTrace, the cylindrical cross section lies in the X-Z plane.
            az = math.radians(self.data_get_number(p_data, "receiver.0.rec_azimuth")) 
            el = math.radians(self.data_get_number(p_data, "receiver.0.rec_elevation")) 
            aim = pysoltrace.Point(math.cos(el)*math.sin(az), math.cos(el)*math.cos(az), math.sin(el))
            element.aim.x = element.position.x + aim.x*1000.
            element.aim.y = element.position.y + aim.y*1000.
            element.aim.z = element.position.z + aim.z*1000.
            
            element.zrot = 0.
            # in the special case of a closed cylinder, use parameters X1=0, X2=0, L = rec height
            element.aperture_params[0] = 0. 
            element.aperture_params[1] = 0. 
            element.aperture_params[2] = self.data_get_number(p_data, "receiver.0.rec_height")
            
            element.aperture = 'l'		#single axis curvature section
            element.surface = 't'
            element.surface_params[0] = 2./diam
            element.interaction = 2
            element.optic = copt
            
            #----------------------
            #close the bottom of the receiver with a circle to prevent internal absorption
            copt = P.add_optic(recname + " spill")
            
            #set the optical properties. This should be a diffuse surface, make it a pillbox distribution w/ equal angular reflection probability.
            copt.front.dist_type = 'g'
            copt.front.reflectivity = 0
            copt.front.slope_error = 100.
            copt.front.spec_error = 100.
            #back
            copt.back.dist_type = 'g'
            copt.back.reflectivity = 0
            copt.back.slope_error = 100.
            copt.back.spec_error = 100.
            #the circle element
            element = r_stage.add_element()
            element.position.x = self.data_get_number(p_data, "receiver.0.rec_offset_x_global")
            element.position.y = self.data_get_number(p_data, "receiver.0.rec_offset_y_global")
            element.position.z = self.data_get_number(p_data, "receiver.0.optical_height") - self.data_get_number(p_data, "receiver.0.rec_height")/2. #optical height includes z offset
                        
            aim = pysoltrace.Point(math.sin(el)*math.cos(az), math.sin(el)*math.sin(az), math.cos(el))
            element.aim.x = element.position.x + aim.x*1000.
            element.aim.y = element.position.y + aim.y*1000.
            element.aim.z = element.position.z + aim.z*1000.

            element.zrot = 0.
            element.aperture_params[0] = diam
            
            element.aperture = 'c'		#single axis curvature section
            element.surface = 'f'
            element.interaction = 2
            element.optic = copt

        elif recgeom in [2,3]:  #CYLINDRICAL_CAV,PLANE_RECT:
            width = self.data_get_number(p_data, "receiver.0.rec_width")
            height = self.data_get_number(p_data, "receiver.0.rec_height")
            #For the elliptical cavity, SolTrace can only handle circular apertures. 
            
            copt.front.dist_type = 'g'
            copt.front.reflectivity = 1.-self.data_get_number(p_data, "receiver.0.absorptance")
            copt.front.slope_error = math.pi/4.
            copt.front.spec_error = math.pi/4.
            copt.back.dist_type = 'g'
            copt.back.reflectivity = 1.-self.data_get_number(p_data, "receiver.0.absorptance")
            copt.back.slope_error = math.pi/4.
            copt.back.spec_error = math.pi/4.
            
            #Add a flat aperture to the stage
            element.enabled = True
            element.position.x = self.data_get_number(p_data, "receiver.0.rec_offset_x_global")
            element.position.y = self.data_get_number(p_data, "receiver.0.rec_offset_y_global")
            element.position.z = self.data_get_number(p_data, "receiver.0.optical_height") #optical height includes z offset
            
            #Calculate the receiver aperture aim point
            az = math.radians(self.data_get_number(p_data, "receiver.0.rec_azimuth")) 
            el = math.radians(self.data_get_number(p_data, "receiver.0.rec_elevation")) 
            aim = pysoltrace.Point(math.cos(el)*math.sin(az), math.cos(el)*math.cos(az), math.sin(el))
            element.aim.x = element.position.x + aim.x*1000.
            element.aim.y = element.position.y + aim.y*1000.
            element.aim.z = element.position.z + aim.z*1000.

            # element->ZRot = R2D*Toolbox::ZRotationTransform(aim);
            element.zrot = P.util_calc_zrot_azel(P.util_calc_unitvect(element.aim))
            
            #Set up the aperture arguments array
            element.aperture_params[0] = width 
            element.aperture_params[1] = height
            #aperture shape 'c' circular or 'r' rectangular
            element.aperture = 'r'
            element.surface = 'f'
            element.interaction = 2
            element.optic = copt

        #Simulation options
        P.num_ray_hits = self.data_get_number(p_data, "fluxsim.0.min_rays") 
        P.max_rays_traced = self.data_get_number(p_data, "fluxsim.0.max_rays")
        seed = self.data_get_number(p_data, "fluxsim.0.seed")
        sun_type = self.data_get_number(p_data, "ambient.0.sun_type")
        P.is_sunshape = self.data_get_number(p_data, "fluxsim.0.is_sunshape_err") and ( sun_type != 0 )  #point sun
        P.is_surface_errors = self.data_get_number(p_data, "fluxsim.0.is_optical_err")
        P.dni = self.data_get_number(p_data, "fluxsim.0.flux_dni")
        

        return P



if __name__ == "__main__":

    pass