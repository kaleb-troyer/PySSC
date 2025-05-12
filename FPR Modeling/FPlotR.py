
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

    def __repr__(self):
        lines = ["Parameters:"]
        for name, value in vars(self).items():
            if isinstance(value, Struct):
                lines.append( 
                    f"  {name:<18}\n\tkey:   {value.key:<20}\n\tunits: {value.units}"
                )
        return "\n".join(lines)

    def _despar(self): 
        self.T_des_i = Struct(key = "T_des_i", dtype = "float64", repr = "Receiver Inlet Temperature",      units = "[C]"   ) 
        self.T_des_o = Struct(key = "T_des_o", dtype = "float64", repr = "Receiver Outlet Temperature",     units = "[C]"   ) 
        self.H_rec   = Struct(key = "H_rec",   dtype = "float64", repr = "Aperture Height",                 units = "[m]"   ) 
        self.W_rec   = Struct(key = "W_rec",   dtype = "float64", repr = "Aperture Width",                  units = "[m]"   ) 
        self.H_ratio = Struct(key = "H_ratio", dtype = "float64", repr = "Curtain / Aperture Height Ratio", units = "[-]"   ) 
        self.W_ratio = Struct(key = "W_ratio", dtype = "float64", repr = "Curtain / Aperture Width Ratio",  units = "[-]"   ) 
        self.D_ratio = Struct(key = "D_ratio", dtype = "float64", repr = "Curtain / Aperture Depth Ratio",  units = "[-]"   ) 
        self.q_des_o = Struct(key = "q_des_o", dtype = "float64", repr = "Receiver Design Output",          units = "[MWt]" ) 

    def _dessol(self):
        self.efficiency    = Struct(key = "eta",                dtype = "float64", repr = "Receiver Efficiency",                    units = "[-]"      )
        self.m_dot_tot     = Struct(key = "m_dot_tot",          dtype = "float64", repr = "Particle Flow Rate",                     units = "[kg/s]"   )
        self.T_sol_o       = Struct(key = "T_particle_hot_rec", dtype = "float64", repr = "Calculated Outlet Temperature",          units = "[C]"      )
        self.q_incident    = Struct(key = "Q_inc",              dtype = "float64", repr = "Total Solar Power Incident on Curtain",  units = "[MWt]"    )
        self.q_reflective  = Struct(key = "Q_refl",             dtype = "float64", repr = "Total Reflection Losses",                units = "[MWt]"    )
        self.q_radiative   = Struct(key = "Q_rad",              dtype = "float64", repr = "Total Radiation Losses",                 units = "[MWt]"    )
        self.q_advective   = Struct(key = "Q_adv",              dtype = "float64", repr = "Total Advection Losses",                 units = "[MWt]"    )
        self.q_conductive  = Struct(key = "Q_cond",             dtype = "float64", repr = "Total Conduction Losses",                units = "[MWt]"    )
        self.q_transport   = Struct(key = "Q_transport",        dtype = "float64", repr = "Total Losses due to Particle Transport", units = "[MWt]"    )
        self.q_sol_o       = Struct(key = "Q_thermal",          dtype = "float64", repr = "Total Power Delivered to Particles",     units = "[MWt]"    )
        self.tauc_avg      = Struct(key = "tauc_avg",           dtype = "float64", repr = "Average Curtain Transmittance",          units = "[-]"      )
        self.rhoc_avg      = Struct(key = "rhoc_avg",           dtype = "float64", repr = "Average Curtain Reflectance",            units = "[-]"      )
        self.qnetc_sol_avg = Struct(key = "qnetc_sol_avg",      dtype = "float64", repr = "Average Net Solar Flux on Curtain",      units = "[MWt/m²]" )
        self.qnetw_sol_avg = Struct(key = "qnetw_sol_avg",      dtype = "float64", repr = "Average Net Solar Flux on Back Wall",    units = "[MWt/m²]" )

class FPlotR(): 
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

    def newparam(self, params: Parameters=None, name: str='', repr: str='', units: str='[-]', operation: str='+', args: list=[]): 
        
        # Creating the parameter
        structure = Struct(
            key=name,
            repr=repr,
            units=units,
            dtype=float,
            default=None,
            value=None
        )

        setattr(params, name, structure)

        # Calculating the data and adding it to the data set
        if structure.key not in self.data_full_set.columns:
            new_col_vals = pd.Series(self.data_full_set[args[0].key])
            for param in args[1:]:
                if param.key not in self.data_full_set.columns:
                    raise KeyError(f"'{param.key}' not found in data_full_set.")

                if callable(operation):
                    new_col_vals = operation(self.data_full_set, *args)
                else:
                    operation = operation.lower().strip()
                    ops = {
                        # Multiplication
                        'mul':         lambda x, y: x * y,
                        'multiply':    lambda x, y: x * y,
                        'multiplication': lambda x, y: x * y,
                        '*':           lambda x, y: x * y,
                        'times':       lambda x, y: x * y,
                        'product':     lambda x, y: x * y,
                        'x':           lambda x, y: x * y,

                        # Division
                        'div':         lambda x, y: x / y,
                        'divide':      lambda x, y: x / y,
                        'division':    lambda x, y: x / y,
                        '/':           lambda x, y: x / y,
                        'quotient':    lambda x, y: x / y,
                        'over':        lambda x, y: x / y,

                        # Addition
                        'add':         lambda x, y: x + y,
                        'addition':    lambda x, y: x + y,
                        '+':           lambda x, y: x + y,
                        'sum':         lambda x, y: x + y,
                        'plus':        lambda x, y: x + y,
                        'increase':    lambda x, y: x + y,

                        # Subtraction
                        'sub':         lambda x, y: x - y,
                        'subtract':    lambda x, y: x - y,
                        'subtraction': lambda x, y: x - y,
                        '-':           lambda x, y: x - y,
                        'minus':       lambda x, y: x - y,
                        'difference':  lambda x, y: x - y,
                        'decrease':    lambda x, y: x - y,
                    }

                    try:
                        new_col_vals = ops[operation](new_col_vals, self.data_full_set[param.key])
                    except KeyError:
                        raise ValueError(f"Unsupported operation: {operation}")

            new_col_vals.name = structure.key
            self.data_full_set = pd.concat([
                self.data_full_set,
                new_col_vals,
            ], axis=1)

        else:
            print(f"'{structure.key}' already exists in data_full_set.")

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

        if isinstance(self.y, list): 
            self.barplot = True

        self._label = label
        self._title = title
        self._line_style = style
        self._plot_case = sum([             #    32 16  8  4  2  1
             (bool(self.x) * self._x),      # 0b  0  0  0  0  0  1
             (bool(self.y) * self._y),      # 0b  0  0  0  0  1  0
             (bool(self.z) * self._z),      # 0b  0  0  0  1  0  0
             (bool(self.c) * self._c),      # 0b  0  0  1  0  0  0
             (bool(self.plot3d) * self._D), # 0b  0  1  0  0  0  0
             (bool(self.barplot) * self._B) # 0b  1  0  0  0  0  0
        ])
        
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
            case 35: # x+y, bar plot 
                self._build_bar()
            case _: 
                raise AttributeError('Plot build configuration not available.')

        if self.legend and not self.barplot: 
            plt.legend()

    def srfit(self, complexity: int=-1, niters: int=40, display: bool=True): 
        '''
        Loads all built data and constructs equations for best fit 
        according to the data.

        ### Parameters
        ---
        title : str=''
            (optional) Defines the displayed title of the visual. 
        '''

        from pysr import PySRRegressor
        from sympy import lambdify

        points = []

        # Check for unexpected lines
        if len(self.ax.lines) > 0:
            raise RuntimeError("Unexpected lines found on the Axes — only scatter plots (collections) are supported.")

        # Extract points from collections
        if isinstance(self.ax, Axes3D):
            for col in self.ax.collections:
                if hasattr(col, '_offsets3d'):
                    x, y, z = col._offsets3d
                    for xi, yi, zi in zip(x, y, z):
                        points.append((xi, yi, zi))
            features = [self.x.key, self.y.key]    

        else: # if the plot is not in 3D
            for col in self.ax.collections:
                offsets = col.get_offsets()
                for offset in offsets:
                    xi, yi = offset
                    points.append((xi, yi))
            features = [self.x.key]    

        points = np.array(points)
        X = pd.DataFrame(points[:, :-1], columns=features)
        Y = points[:,  -1]

        # Example: fitting to sample data
        model = PySRRegressor(
            model_selection="best",     # Pick best model balancing simplicity & accuracy
            niterations=niters,         # Number of evolutionary steps
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "exp", "sin", "cos"],
            extra_sympy_mappings={"square": lambda x: x**2},
            verbosity=1,
            select_k_features=3 
        )

        model.fit(X, Y)

        if display and isinstance(self.ax, Axes3D): 
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

            # Prepare meshgrid for plotting surface
            x_lin = np.linspace(x.min(), x.max(), 100)
            y_lin = np.linspace(y.min(), y.max(), 100)
            x_grid, y_grid = np.meshgrid(x_lin, y_lin)
            xymesh = np.column_stack([x_grid.ravel(), y_grid.ravel()])

            # Predict using specified equation
            if complexity == 'best': 
                z_prediction = model.predict(xymesh).reshape(x_grid.shape)
            else: # getting specific solution based on complexity
                vars = model.feature_names_in_ 
                if complexity > len(model.equations_): 
                    complexity == len(model.equations_)
                cell = model.equations_.iloc[complexity]
                expr = cell["sympy_format"]
                f = lambdify(vars, expr, modules="numpy")
                z_prediction = f(*xymesh.T).reshape(x_grid.shape)

            # Plot surface from symbolic regression
            self.ax.plot_surface(x_grid, y_grid, z_prediction, color='red', alpha=0.5, label='Best Fit')

        elif display: 
            x = points[:, 0]
            y = points[:, 1]

            # Prepare linearly spaced x values for prediction
            x_lin = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)

            # Predict using specified equation
            if complexity == 'best':
                y_prediction = model.predict(x_lin)
            else:
                vars = model.feature_names_in_
                if complexity >= len(model.equations_):
                    complexity = len(model.equations_) - 1
                cell = model.equations_.iloc[complexity]
                expr = cell["sympy_format"]
                f = lambdify(vars, expr, modules="numpy")
                y_prediction = f(x_lin.T[0])  # x_lin is 2D; convert to 1D

            # Plot regression line
            self.ax.plot(x_lin, y_prediction, color='red', alpha=0.5, label='Best Fit')

        if complexity == 'best': 
            equation = model.get_best()
        else: equation = model.equations_.iloc[complexity]

        return equation

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

        # checking if the figure is already in 3D
        for ax in self.fig.axes:
            if isinstance(ax, Axes3D):
                self.ax = ax
                break
        else: # if not in 3D, creating a 3D projection
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

    def _build_base_bar(self): 

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
        self.ax.set_ylabel(f"Receiver Losses")
        self.ax.set_title(self._title)

        self.ax.set_xlim(-0.5, len(self.data) - 0.5)
        self.ax.yaxis.grid(True, which='major', color='lightgray')
        self.ax.xaxis.grid(False)
        for spine in ['top', 'right', 'left']:
            self.ax.spines[spine].set_visible(False)
        self.ax.spines['bottom'].set_color('lightgray')

        self.ax.tick_params(axis='both', which='both', length=0)

        self.ax.tick_params(labelsize=9, colors='dimgray')
        self.ax.xaxis.label.set_color('dimgray')
        self.ax.yaxis.label.set_color('dimgray')
        self.ax.title.set_fontsize(11)
        self.ax.title.set_color('dimgray')

    def _build_bar(self):
        self._getdata()
        self._build_base_bar()

        colors  = sns.color_palette("mako", len(self.y))
        labels  = self.data[self.x.key]

        bottom  = [0] * len(self.data)
        indices = list(range(len(self.data)))
        for color, param in zip(colors, self.y):
            self.ax.bar(indices, self.data[param.key], bottom=bottom, color=color, label=param.repr, zorder=3, width=0.5)
            bottom = [i + j for i, j in zip(bottom, self.data[param.key])]

        self.ax.set_xticks(indices)
        self.ax.set_xticklabels([f'{x:.2f}' for x in labels], rotation=0)

        if self.legend:
            legend = self.ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),  # centered below the axis
                ncol=int(len(self.y)/2),      # horizontal layout
                frameon=False                 # no box around the legend
            )

            for text in legend.get_texts():
                text.set_color('dimgray')
                text.set_fontsize(9)

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

        if self.colorbar: 
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
        if self.scatter and self.colorbar: 
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
        self._x = 0b000001  # x-axis key
        self._y = 0b000010  # y-axis key
        self._z = 0b000100  # z-axis key
        self._c = 0b001000  # c-axis key
        self._D = 0b010000  # 3D plot option
        self._B = 0b100000  # Bar plot option

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
        self.barplot = False
        self.colorbar = False

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

    source = os.path.join(os.getcwd(), 'FPR Modeling', 'results', '2025-05-05_solutions.csv')

    params = Parameters()
    dtypes = {par.key: par.dtype for par in params.get()}
    fprplt = FPlotR(source, dtypes=dtypes)

    fprplt.newparam(
        params, 'dT', 'Receiver Temperature Change', '[C]', '-', [params.T_des_o, params.T_des_i] 
    )

    fprplt.x = params.T_des_o_K
    fprplt.y = params.T_des_i_K
    fprplt.z = params.efficiency

    fprplt.legend = False 
    fprplt.plot3d = True
    fprplt.scatter = True
    fprplt.colorbar = False
    fprplt.grayscale = False 
    fprplt.linelabels = False 

    fprplt.filter(
        (params.T_des_i, lambda x: x == 550),
        (params.q_des_o, lambda x: x == 200), 
        (params.T_des_o, (max, params.efficiency)) 
    ) 

    fprplt.build()
    fprplt.show()
    
    def case1(): 
        fprplt.x = params.q_des_o
        fprplt.y = params.T_des_i
        fprplt.z = params.efficiency
        fprplt.c = params.T_des_o

        fprplt.legend = False 
        fprplt.plot3d = True
        fprplt.scatter = True
        fprplt.colorbar = False
        fprplt.grayscale = False 
        fprplt.linelabels = False 

        for val in fprplt.data_full_set[params.T_des_o.key].unique():

            for par in fprplt.data_full_set[params.T_des_i.key].unique(): 

                if par == fprplt.data_full_set[params.T_des_i.key].unique()[-1] and val == fprplt.data_full_set[params.T_des_o.key].unique()[-1]: 
                    fprplt.colorbar = True

                if val > par: 
                    fprplt.filter(
                        (params.T_des_o, lambda x: x == val), 
                        (params.T_des_i, lambda x: x == par),
                        (params.q_des_o, (max, params.efficiency)) 
                    ) 

                    fprplt.build()
        
        fprplt.show()
    def case2(): 
        fprplt.x = params.T_des_o
        fprplt.y = params.T_des_i
        fprplt.z = params.efficiency

        fprplt.legend = False 
        fprplt.plot3d = True
        fprplt.scatter = True
        fprplt.colorbar = False
        fprplt.grayscale = False 
        fprplt.linelabels = False 

        for par in fprplt.data_full_set[params.T_des_i.key].unique(): 

            fprplt.filter(
                (params.T_des_i, lambda x: x == par),
                (params.q_des_o, lambda x: x == 200), 
                (params.T_des_o, (max, params.efficiency)) 
            ) 

            fprplt.build()

        solution = fprplt.srfit(complexity=6)
        print(solution['equation'])

        fprplt.show()
    def case3(): 
        def funct(df, To, Ti): 
            return np.cos((np.log(df[To.key]) - (df[Ti.key] * 0.28989363)) / df[To.key]) - 0.24336664

        fprplt.newparam(
            params, 'eta_prime', 'Predicted Performance', '[-]', funct, [params.T_des_o, params.T_des_i] 
        )

        fprplt.x = params.q_des_o
        fprplt.y = params.eta_prime
        fprplt.z = params.efficiency

        fprplt.legend = False 
        fprplt.plot3d = True
        fprplt.scatter = True
        fprplt.colorbar = False
        fprplt.grayscale = False 
        fprplt.linelabels = False 

        for val in fprplt.data_full_set[params.T_des_i.key].unique(): 

            fprplt.filter(
                (params.T_des_i, lambda x: x == val),
                (params.q_des_o, (max, params.efficiency)) 
            ) 

            fprplt.build()

        solution = fprplt.srfit(complexity='best')
        print(solution['equation'])

        fprplt.show()
    def case4(): 
        def get_thickness(df, mdot, Wa, Wr): 

            rho = 2400
            sv = 0.01
            v = 6

            th = df[mdot.key] / (df[Wa.key] * df[Wr.key] * v * rho * sv)
            return th

        fprplt.newparam(
            params, 'th', 'Approximate Curtain Thickness', '[m]', get_thickness, [
                params.m_dot_tot, params.W_rec, params.W_rec
            ]
        )

        fprplt.x = params.T_des_o
        fprplt.y = params.T_des_i
        fprplt.z = params.th

        fprplt.legend = False 
        fprplt.plot3d = True
        fprplt.scatter = True
        fprplt.colorbar = False
        fprplt.grayscale = False 
        fprplt.linelabels = False 

        for par in fprplt.data_full_set[params.T_des_i.key].unique(): 

            fprplt.filter(
                (params.T_des_i, lambda x: x == par),
                (params.q_des_o, lambda x: x == 200), 
                (params.T_des_o, (max, params.efficiency)) 
            ) 

            fprplt.build()

        fprplt.show()
    def case5(): 
        def toKelvin(df, T, _):
            return df[T.key] + 273.15

        fprplt.newparam(
            params, 'T_des_o_K', 'Receiver Outlet Temp', '[K]', toKelvin, [params.T_des_o, params.T_des_o]
        )

        fprplt.newparam(
            params, 'T_des_i_K', 'Receiver Inlet Temp', '[K]', toKelvin, [params.T_des_i, params.T_des_i]
        ) 

        fprplt.normalize(
            params, params.T_des_o_K, 30 + 273.15
        )

        fprplt.normalize(
            params, params.T_des_i_K, 30 + 273.15
        )

        fprplt.x = params.T_des_o_K_norm
        fprplt.y = params.T_des_i_K_norm
        fprplt.z = params.efficiency

        fprplt.legend = False 
        fprplt.plot3d = True
        fprplt.scatter = True
        fprplt.colorbar = False
        fprplt.grayscale = False 
        fprplt.linelabels = False 

        for par in fprplt.data_full_set[params.T_des_i.key].unique(): 

            fprplt.filter(
                (params.T_des_i, lambda x: x == par),
                (params.q_des_o, lambda x: x == 200), 
                (params.T_des_o, (max, params.efficiency)) 
            ) 

            fprplt.build()

        solution = fprplt.srfit(complexity=6)
        print(solution['equation'])

        fprplt.show()

