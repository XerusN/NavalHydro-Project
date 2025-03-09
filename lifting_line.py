import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from utils import *

# Unless otherwise specified all units are SI

class PropellerCharacteristics:
    d: float
    z: float
    p_d_07: float
    balde_ar: float
    def __init__(self, d, z, p_d_07, blade_ar):
        self.d = d
        self.z = z
        self.p_d_07 = p_d_07
        self.balde_ar = blade_ar
    
class Fluid:
    density: float
    kin_visc: float
    vapor_p: float
    p_atm: float
    def __init__(self, density, kin_visc, vapor_p, p_atm):
        self.density = density
        self.kin_visc = kin_visc
        self.vapor_p = vapor_p
        self.p_atm = p_atm

class Geometry:
    r_R: np.ndarray
    c_d: np.ndarray
    t_d: np.ndarray
    p_d: np.ndarray
    id_07: int
    def __init__(self, r_R, c_d, t_d, p_d, id_07):
        self.r_R = r_R
        self.c_d = c_d
        self.t_d = t_d
        self.p_d = p_d
        self.id_07 = id_07

# Used to import precomputed a_0 values with xfoil
def import_a0(path: str) -> np.array:
    data = np.loadtxt(path)
    return data[:, 1]

# Linear foil theory
# Expects angles in radian
def cl_from_a(a: float, a0: float):
    return 2*pi*(a - a0)


# n and V from Reynolds and j?
def n_v_from_j_rn(j: float, rn07: float, water: Fluid, prop_c: PropellerCharacteristics, geo: Geometry) -> float:
    n = rn07*geo.c_d[geo.id_07]*prop_c.d**2/water.kin_visc/np.sqrt(j**2+0.7**2/(2*pi)**2)
    v = j*n*prop_c.d
    return n, v
    
def __main__():
    rn = 9.78e7
    prop_c = PropellerCharacteristics