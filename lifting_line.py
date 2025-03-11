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

class Propeller:
    geo: Geometry
    char: PropellerCharacteristics
    def __init__(self, geo, char):
        self.char = char
        self.geo = geo

# Used to import precomputed a_0 values with xfoil
def import_a0(path: str) -> np.array:
    data = np.loadtxt(path)
    return data[:, 1]

def import_geometry(path: list) -> Geometry:
    data = np.loadtxt(path, skiprows=1, delimiter=',')
    for i in range(len(data[:, 0])):
        if data[i, 0] == 0.7:
            id_07 = i
            return Geometry(data[:, 0], data[:, 1], data[:, 2], data[:, 3], i)
    print('No value at r/R = 0.7')
    exit
    
# Linear foil theory
# Expects angles in radian
def cl_from_aoa(a: float, a0: float):
    return 2*pi*(a - a0)


# n and V from Reynolds and j?
def n_v_from_j_rn(j: float, rn07: float, water: Fluid, prop: Propeller) -> float:
    n = rn07*prop.geo.c_d[prop.geo.id_07]/prop.char.d**2*water.kin_visc/np.sqrt(j**2+0.7**2/(2*pi)**2)
    v = j*n*prop.char.d
    return n, v

def geo_pitch_angle(prop: Propeller, section: int):
    return np.arctan(prop.geo.p_d[section]/(pi*prop.geo.r_R[section]))    # Lecture 12-8

def beta_mean(n: float, v: float, prop: Propeller, section: int):
    return np.arctan(v/(prop.geo.r_R[section]*prop.char.d*pi*n))

def aoa(beta: float, prop: Propeller, section: int):
    theta = geo_pitch_angle(prop, section)
    return theta - beta

def rn(v: float, l: float, nu: float):
    return v*l/nu

def cf(v_inf: float, prop: Propeller, water: Fluid, section: int):
    print(rn(v_inf, prop.geo.c_d[section]*prop.char.d, water.kin_visc))
    return 0.075/(np.log10(rn(v_inf, prop.geo.c_d[section]*prop.char.d, water.kin_visc))-2.)**2

def cd_ittc(v_inf: float, prop: Propeller, water: Fluid, section: int):
    return 2.*cf(v_inf, prop, water, section)*(1. + 2.*prop.geo.t_d[section]/prop.geo.c_d[section])

def dr_section(prop: Propeller, section: int):
    if section == len(prop.geo.r_R) - 2:
        return (prop.geo.r_R[section + 1] - prop.geo.r_R[section])*prop.char.d/2.
    if section == 0:
        return (prop.geo.r_R[section+1] - prop.geo.r_R[section])/2.*prop.char.d/2.
    return ((prop.geo.r_R[section+1] - prop.geo.r_R[section])/2. + (prop.geo.r_R[section] - prop.geo.r_R[section-1])/2.)*prop.char.d/2.
    
    
def q2():
    rn07 = 9.78e7
    prop_c = PropellerCharacteristics(4.65, 4, 1.1, 0.65)
    water = Fluid(1025, 1.08e-6, 2160, 101325)
    geo = import_geometry("../../ProvidedFiles/Geometry.txt")

    prop = Propeller(geo, prop_c)

    j = np.linspace(0.4, 1.0, 7)

    n, v = n_v_from_j_rn(j, rn07, water, prop)

    T = np.zeros(len(n))
    Q = np.zeros(len(n))
    # Iterate through the blade sections
    for section in range(len(prop.geo.r_R)-1):
        beta = beta_mean(n, v, prop, section)
        aoa_ = aoa(beta, prop, section)
        #Fixed value of a0 for test purpose
        cl = cl_from_aoa(aoa_, 0.)
        v_inf = np.sqrt(v**2 + (prop.geo.r_R[section]*prop.char.d*pi*n)**2)
        cd = cd_ittc(v_inf, prop, water, section)
        dr = dr_section(prop, section)
        dD = water.density/2.*v_inf**2*cd*prop.geo.c_d[section]*prop.char.d*dr
        dL = water.density/2.*v_inf**2*cl*prop.geo.c_d[section]*prop.char.d*dr
        
        dT = dL*np.cos(beta) - dD*np.sin(beta)
        dQ = prop.geo.r_R[section]*prop.char.d*(dL*np.sin(beta) + dD*np.cos(beta))
        T += dT
        Q += dQ
        
    kt = T /(water.density * n**2 * prop.char.d**4)
    kq = Q /(water.density * n**2 * prop.char.d**5)

    export = []
    for i in range(len(j)):
        export.append([j[i], kt[i], kq[i]])
    export = np.array(export)

    np.savetxt('q2.txt', export, header='j  kt  kq')

q2()