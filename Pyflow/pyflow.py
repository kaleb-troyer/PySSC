
from pyfluids import Fluid, FluidsList, Input
from enum import Enum
import numpy as np
import atexit
import ctypes
import math
import sys
import os

class Geometry(Enum): 
    pipe    = 10
    duct    = 20
    annulus = 30
    custom  = 100

class Model(): 

    def __init__(self):
        self.geometry  = None
        self.diameter  = None
        self.length    = None
        self.area      = None
        self.dims      = None
        self.roughness = None
        self.relrough  = None
        self.velocity  = None
        self.massflow  = None

    @classmethod
    def Pipe(cls, D, L: float=float('inf'), v: float=None, massflow: float=None, e: float=0) -> 'Model': 
        """
        Creates a pipe model with specified dimensions and flow properties.

        This method packages pipe dimensions for use by Pyflow and returns 
        an instance of the `Model` class.

        Args:
            D (float): Inner or wetted diameter of the pipe (meters).
            L (float, optional): Length of the pipe (meters). Defaults to infinity.
            v (float, optional): Bulk fluid velocity through the pipe (m/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            massflow (float, optional): Mass flow rate through the pipe (kg/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            e (float, optional): Absolute roughness of the pipe wall (meters). 
                Relative roughness is calculated using the hydraulic diameter.

        Returns:
            Model: An instance of the `Model` class with the specified properties.
        """
        instance = cls()

        instance.geometry = Geometry.pipe 
        instance.diameter = D 
        instance.length = L
        instance.area = (np.pi / 4) * D**2 
        instance.dims = (D, 0, L) 
        instance.roughness = e
        instance.relrough = e / instance.diameter
        instance.velocity = v 
        instance.massflow = massflow

        instance._validation()
        return instance

    @classmethod
    def Duct(cls, H, W, L: float=float('inf'), v: float=None, massflow: float=None, e: float=0) -> 'Model': 
        """
        Creates a duct model with specified dimensions and flow properties.

        This method packages duct dimensions for use by Pyflow and returns 
        an instance of the `Model` class.

        Args:
            H (float): Height of the duct (meters).
            W (float): Width of the duct (meters).
            L (float, optional): Length of the duct (meters). Defaults to infinity.
            v (float, optional): Bulk fluid velocity through the duct (m/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            massflow (float, optional): Mass flow rate through the duct (kg/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            e (float, optional): Absolute roughness of the duct wall (meters). 
                Relative roughness is calculated using the hydraulic diameter.

        Returns:
            Model: An instance of the `Model` class with the specified properties.
        """
        instance = cls()

        instance.geometry = Geometry.duct 
        instance.diameter = (4 * H * W) / (2 * (H + W)) 
        instance.length = L
        instance.area = H * W 
        instance.dims = (H, W, L) 
        instance.roughness = e
        instance.relrough = e / instance.diameter
        instance.velocity = v 
        instance.massflow = massflow

        instance._validation()
        return instance

    @classmethod
    def Annulus(cls, Do, Di, L: float=float('inf'), v: float=None, massflow: float=None, e: float=0): 
        """
        Creates an annulus model with specified dimensions and flow properties.

        This method packages annulus dimensions for use by Pyflow and returns 
        an instance of the `Model` class.

        Args:
            Do (float): Inner diameter of the outer pipe (meters).
            Di (float): Outer wetted diameter of the inner pipe (meters).
            L (float, optional): Length of the annulus (meters). Defaults to infinity.
            v (float, optional): Bulk fluid velocity through the annulus (m/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            massflow (float, optional): Mass flow rate through the annulus (kg/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            e (float, optional): Absolute roughness of the annulus wall (meters). 
                Relative roughness is calculated using the hydraulic diameter.

        Returns:
            Model: An instance of the `Model` class with the specified properties.
        """
        instance = cls()

        instance.geometry = Geometry.annulus 
        instance.diameter = 4 * (np.pi / 4) * (Do**2 - Di**2) / (np.pi * (Di + Do)) 
        instance.length = L
        instance.area = (np.pi / 4) * (Do**2 - Di**2) 
        instance.dims = (Do, Di, L) 
        instance.roughness = e
        instance.relrough = e / instance.diameter
        instance.velocity = v 
        instance.massflow = massflow

        instance._validation()
        return instance

    @classmethod
    def Custom(cls, Dh, L: float=float('inf'), v: float=None, massflow: float=None, e: float=0): 
        """
        Creates a custom geometry model with specified dimensions and flow properties.

        **Note:** This implementation is incomplete. Future versions may use an image 
        with black-and-white pixels to define the 2D geometry and calculate 
        the expected heat transfer coefficient.

        Args:
            Dh (float): Hydraulic diameter of the custom geometry (meters).
            L (float, optional): Length of the geometry (meters). Defaults to infinity.
            v (float, optional): Bulk fluid velocity through the geometry (m/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            massflow (float, optional): Mass flow rate through the geometry (kg/s). 
                If not manually set, it is automatically updated based on changes 
                to geometry and fluid properties.
            e (float, optional): Absolute roughness of the surface (meters). 
                Relative roughness is calculated using the hydraulic diameter.

        Returns:
            Model: An instance of the `Model` class with the specified properties.
        """
        instance = cls()

        instance.geometry = Geometry.custom 
        instance.diameter = Dh 
        instance.length = L 
        instance.area = np.pi * (Dh / 2)**2 
        instance.dims = (Dh, 0, L) 
        instance.roughness = e
        instance.relrough = e / instance.diameter
        instance.velocity = v 
        instance.massflow = massflow

        instance._validation()
        return instance

    def _validation(self):

        for var_name, value in self.__dict__.items():
            if callable(value) or var_name == 'geometry': 
                continue
            
            if isinstance(value, (float, int)) and value < 0:
                raise ValueError(f"Invalid value for {var_name}: {value}. All physical dimensions and properties must be positive.")

class Pyflow(): 
    """
    Pyflow class for modeling heat transfer and fluid flow in various geometries.

    This class allows users to define pipe, duct, or annulus geometries along with 
    fluid properties to calculate relevant parameters such as the heat transfer 
    coefficient (HTC) and Reynolds number. The flow conditions are automatically updated 
    based on changes to the geometry and fluid state, unless explicitly set by the user.

    This class relies on the `Fluid` object from pyfluids to retrieve fluid properties. 
    Since the `Fluid` object is mutable, any external updates to it are automatically 
    reflected in Pyflow. When 'Pyflow.update()' is called, the latest fluid properties 
    are incorporated into the calculations.

    Example:
        ```python
        from pyfluids import Fluid, FluidsList, Input

        pipe = Pyflow(
            model.Pipe(D=0.2, L=20, v=10), 
            Fluid(Fluidslist.Water).with_state(
                Input.temperature(100), Input.pressure(101325)
            )
        )

        pipe.htc  
        # Returns the heat transfer coefficient.

        # The velocity is recalculated for every successive geometry 
        # and fluid state, so it does not need to be provided again. 
        pipe.update(model.Pipe(D=0.1, L=20))

        pipe.htc  
        # Returns the updated heat transfer coefficient.
        ```

    """

    def __init__(self, model: Model=None, fluid: Fluid=None, quiet: bool=False):

        # loading the pyflow DLL
        path = os.path.join(os.getcwd(), 'Pyflow')
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            self.pdll = ctypes.CDLL(os.path.join(path, "htc-pyflow.dll"))
        elif sys.platform == 'darwin':
            self.pdll = ctypes.CDLL(os.path.join(path, "htc-pyflow.dylib"))  # Never tested
        elif sys.platform.startswith('linux'):
            self.pdll = ctypes.CDLL(os.path.join(path, "htc-pyflow.so"))  # Never tested
        else: print( 'Platform not supported ', sys.platform)

        # class variables
        self.quiet = quiet
        self._model = None
        self._fluid = None
        self._last_velocity = None
        self._last_massflow = None
        self._last_density = None
        self._last_area = None
        self._reynolds = None
        self._nusselt = None
        self._friction_factor = None

        # Establishing function argtypes and return types
        self._pdll_types()

        # Updating parameters and calculating the solution
        self.update(model, fluid)

        # Ensuring memory cleanup on process close
        atexit.register(lambda: self.pdll.DeleteFlowCalculator(self._flowcalc))

    def __repr__(self):
        prel = 10
        posl = 25
        return (
            f"{'dx':.<{prel}}{float(self._model.length):.>{posl}.2f} [m]\n"
            f"{'u':.<{prel}}{float(self._model.velocity):.>{posl}.2f} [m/s]\n"
            f"{'Re':.<{prel}}{self.Re:.>{posl}.3e} [-]\n"
            f"{'ff':.<{prel}}{self.ff:.>{posl}.4f} [-]\n"
            f"{'Nu':.<{prel}}{self.Nu:.>{posl}.2f} [-]\n"
            f"{'htc':.<{prel}}{self.htc:.>{posl}.2f} [W/m2-K]\n"
            f"{'dP':.<{prel}}{self.dp/1000:.>{posl}.2f} [kPa]"
        )

    def update(self, model: Model=None, fluid: Fluid=None): 
        """
        Updates the Pyflow class with new model or fluid parameters. Then, the 
        flow conditions are recalculated. 

        ### Attributes  

        ---
        **model**: *pyflow* `Model`     
        -> Pyflow model geometry and dimensions class.  
        **fluid**: *pyfluids* `Fluid`  
        -> Pyfluids fluid state and properties class.  
        """

        # loading the first argument, if there is one
        if model is None: 
            pass
        elif model and isinstance(model, Model): 
            self._model = model
        elif isinstance(model, Fluid): 
            self._fluid = model
        else: raise ValueError("Invalid type for model argument.")

        # loading the second argument, if there is one
        if fluid is None: 
            pass
        elif fluid and isinstance(fluid, Fluid): 
            self._fluid = fluid
        elif isinstance(fluid, Model): 
            self._model = fluid
        else: raise ValueError("Invalid type for fluid argument.")

        # updating the mass flow rate and fluid velocity
        if not self._model or not self._fluid: 
            # verifying that the model and fluid have been specified
            raise ValueError("The model and fluid must be specified on class creation.")
        elif model and model.massflow is None and model.velocity is not None: 
            # calculating mass flow based on specified fluid velocity, fluid density
            self._model.massflow = self._model.velocity * (self._fluid.density * self._model.area)
        elif model and model.velocity is None and model.massflow is not None: 
            # calculating fluid velocity based on specified mass flow, fluid density 
            self._model.velocity = self._model.massflow / (self._fluid.density * self._model.area)
        elif not model or (model.massflow is None and model.velocity is None): 
            # updating the fluid velocity based on condition changes, verifying mass flow doesn't change
            self._model.velocity = (
                (self._last_velocity * self._last_density * self._last_area)
                 / (self._fluid.density * self._model.area)
            ) 

            self._model.massflow = self._model.velocity * (self._fluid.density * self._model.area)
            assert math.isclose(self._model.massflow, self._last_massflow, rel_tol=1e-8)
        else: raise ValueError("Specify either mass flow rate or fluid velocity, not both.")

        # storing current config data so velocity can be automatically computed when
        # the geometry or fluid is updated, based on conservation of mass. 
        self._last_velocity = self._model.velocity
        self._last_massflow = self._model.massflow
        self._last_density  = self._fluid.density
        self._last_area     = self._model.area

        # verifying specified design values are nonzero and positive. 
        if self._last_area <= 0.0:
            raise ValueError("Cross-sectional area cannot be zero or negative.")
        elif self._last_velocity <= 0.0:
            raise ValueError("Fluid velocity cannot be zero or negative.")
        elif self._last_massflow <= 0.0:
            raise ValueError("Mass flow rate cannot be zero or negative.")
        elif self._last_density <= 0.0:
            raise ValueError("Fluid density cannot be zero or negative.")

        # Constructing the flow calculator 
        if not hasattr(self, '_flowcalc'): 
            self._flowcalc = self.pdll.CreateFlowCalculator(
                self._model.area, 
                self._model.diameter, 
                self._model.relrough, 
                self._model.velocity, 
                self._fluid.prandtl, 
                self._fluid.density, 
                self._fluid.dynamic_viscosity, 
                self._fluid.conductivity, 
            )
        else: self.pdll.updater(
                self._flowcalc, 
                self._model.area, 
                self._model.diameter, 
                self._model.relrough, 
                self._model.velocity, 
                self._fluid.prandtl, 
                self._fluid.density, 
                self._fluid.dynamic_viscosity, 
                self._fluid.conductivity, 
            )

        # Running the flow calculator
        assert self.pdll.core(
            self._flowcalc, 
            self._model.geometry.value, 
            ctypes.cast(
                (ctypes.c_double * 3)(*self._model.dims), ctypes.POINTER(ctypes.c_double)
            ), 
        )

        # retreiving the design solution
        self._reynolds = self.pdll.getReynolds(self._flowcalc)
        self._nusselt = self.pdll.getNusselt(self._flowcalc)
        self._friction_factor = self.pdll.getFrictionFactor(self._flowcalc)
        self._heat_transfer_coefficient = self.pdll.getHeatTransferCoefficient(self._flowcalc)
        self._pressure_drop = self.pdll.getPressureDrop(self._flowcalc)

        # checking the integrity of the solution
        self._check_warnings()

    def _check_warnings(self): 

        warning_messages = [
            f'Invalid geometry code ({self._model.geometry}).', 
            f'Reynolds number ({self.Re:.4f}) is too small or negative (Re < 0.001).', 
            f'Reynolds number ({self.Re:.2e}) is too large (Re > 5.0e+06).', 
            f'Prandtl number ({self._fluid.prandtl:.2f}) is outside of the valid range. For turbulent flow, 0.004 < Pr < 2000. For laminar flow, Pr > 0.1.', 
            f'L/D ratio ({self._model.length / self._model.diameter:.2f}) should be greater than 1.', 
            f'Relative roughness ({self._model.relrough:.4f}) should be < 0.05 and positive.', 
            f'Graetz number is outside the acceptable range for a laminar flow calculation.', 
        ]

        self._warnings = self.pdll.getWarnings(self._flowcalc)
        self._binflags = bin(self._warnings)[2:].zfill(32)
        for i, flag in enumerate(self._binflags[::-1]): 
            if flag == '1' and not self.quiet: 
                if i < len(warning_messages): 
                    print('WARNING:', warning_messages[i])
                else: print(f"Uknown warning, loc[{i}]=1")

    def _pdll_types(self): 
        self.pdll.core.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.pdll.core.restype = ctypes.c_bool
        self.pdll.CreateFlowCalculator.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.pdll.CreateFlowCalculator.restype = ctypes.c_void_p
        self.pdll.DeleteFlowCalculator.argtypes = [ctypes.c_void_p]
        self.pdll.DeleteFlowCalculator.restype = None
        self.pdll.updater.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.pdll.updater.restype = ctypes.c_bool

        self.pdll.getNusselt.argtypes = [ctypes.c_void_p]
        self.pdll.getNusselt.restype = ctypes.c_double
        self.pdll.getReynolds.argtypes = [ctypes.c_void_p]
        self.pdll.getReynolds.restype = ctypes.c_double
        self.pdll.getPressureDrop.argtypes = [ctypes.c_void_p]
        self.pdll.getPressureDrop.restype = ctypes.c_double
        self.pdll.getFrictionFactor.argtypes = [ctypes.c_void_p]
        self.pdll.getFrictionFactor.restype = ctypes.c_double
        self.pdll.getHeatTransferCoefficient.argtypes = [ctypes.c_void_p]
        self.pdll.getHeatTransferCoefficient.restype = ctypes.c_double

        self.pdll.getWarnings.argtypes = [ctypes.c_void_p]
        self.pdll.getWarnings.restype = ctypes.c_int 

    #---Property Managers
    @property
    def Reynolds(self):
        return self._reynolds
    @Reynolds.setter
    def Reynolds(self, value): 
        self._reynolds = value

    @property
    def Re(self):
        return self._reynolds
    @Re.setter
    def Re(self, value): 
        self._reynolds = value     

    @property
    def Nusselt(self):
        return self._nusselt
    @Nusselt.setter
    def Nusselt(self, value): 
        self._nusselt = value

    @property
    def Nu(self):
        return self._nusselt
    @Nu.setter
    def Nu(self, value): 
        self._nusselt = value    

    @property
    def friction_factor(self):
        return self._friction_factor
    @friction_factor.setter
    def friction_factor(self, value): 
        self._friction_factor = value
        
    @property
    def ff(self):
        return self._friction_factor
    @ff.setter
    def ff(self, value): 
        self._friction_factor = value    

    @property
    def heat_transfer_coefficient(self):
        return self._heat_transfer_coefficient
    @heat_transfer_coefficient.setter
    def heat_transfer_coefficient(self, value): 
        self._heat_transfer_coefficient = value
        
    @property
    def htc(self):
        return self._heat_transfer_coefficient
    @htc.setter
    def htc(self, value): 
        self._heat_transfer_coefficient = value   

    @property
    def pressure_drop(self):
        return self._pressure_drop
    @pressure_drop.setter
    def pressure_drop(self, value): 
        self._pressure_drop = value
        
    @property
    def dp(self):
        return self._pressure_drop
    @dp.setter
    def dp(self, value): 
        self._pressure_drop = value 

if __name__=='__main__': 

    fluid = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(648), Input.pressure(25e6)
    )

    model = Pyflow(
        Model.Pipe(D=0.003, L=0.6, v=10, e=5e-4), 
        fluid
    )

    print(model)


