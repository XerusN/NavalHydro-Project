import numpy as np
import matplotlib.pyplot as plt
from InductionFactors import *
from SingularIntegration import *
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
    # Index of the section with r/R = 0.7
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

# Used to import precomputed cl values with xfoil
def import_cl(path) -> np.array:
    cl_a = []
    data = np.loadtxt(path+'results_167.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_200.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_300.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_350.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_400.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_450.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_500.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_550.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_600.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_700.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_800.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    data = np.loadtxt(path+'results_900.txt', skiprows=2, delimiter = '|')
    cl_a.append(data)
    return np.array(cl_a)

# Imports the geometry
def import_geometry(path: list) -> Geometry:
    data = np.loadtxt(path, skiprows=1, delimiter=',')
    for i in range(len(data[:, 0])):
        if data[i, 0] == 0.7:
            id_07 = i
            return Geometry(data[:, 0], data[:, 1], data[:, 2], data[:, 3], i)
    print('No value at r/R = 0.7')
    exit
    
# Interpolate Lift coefficient in the xfoil data
def cl_from_xfoil(a: float, cl_a, section: int):
    cl = np.interp(rad_to_deg(a), cl_a[section, :, 0], cl_a[section, :, 1])
    return cl

# n and V from prescribed Reynolds and j
def n_v_from_j_rn(j: float, rn07: float, water: Fluid, prop: Propeller):
    n = rn07/prop.geo.c_d[prop.geo.id_07]/prop.char.d**2*water.kin_visc/np.sqrt(j**2+0.7**2*pi**2)
    v = j*n*prop.char.d
    return n, v

def geo_pitch_angle(prop: Propeller, section: int):
    return np.arctan(prop.geo.p_d[section]/(pi*prop.geo.r_R[section]))    # Lecture 12-8

# hydrodynamic pitch angle without induced velocities
def beta_mean(n: float, v: float, prop: Propeller, section: int):       #Correction exercise 6
    return np.arctan(v/(prop.geo.r_R[section]*prop.char.d*pi*n))

# computes the AoA from geometry and hydrodynamic pitch angle
def aoa(beta: float, prop: Propeller, section: int):
    theta = geo_pitch_angle(prop, section)
    return theta - beta

# Computes the reynolds number
def rn(v: float, l: float, nu: float):
    return v*l/nu

# Friction coefficient (ITTC correlation)
def cf(v_inf: float, prop: Propeller, water: Fluid, section: int):
    return 0.075/(np.log10(rn(v_inf, prop.geo.c_d[section]*prop.char.d, water.kin_visc))-2.)**2

# Drag coefficient
def cd_ittc(v_inf: float, prop: Propeller, water: Fluid, section: int):
    return 2.*cf(v_inf, prop, water, section)*(1. + 2.*prop.geo.t_d[section]/prop.geo.c_d[section])

# Computes the span of the section for integration purpose
def dr_section(prop: Propeller, section: int):
    if section == len(prop.geo.r_R) - 2:
        return (prop.geo.r_R[section + 1] - prop.geo.r_R[section-1])*3./4.*prop.char.d/2.
    if section == 0:
        return (prop.geo.r_R[section+1] - prop.geo.r_R[section])/2.*prop.char.d/2.
    return ((prop.geo.r_R[section+1] - prop.geo.r_R[section])/2. + (prop.geo.r_R[section] - prop.geo.r_R[section-1])/2.)*prop.char.d/2.

def export(filename: str, j, kt, kq, eta):
    export = []
    for i in range(len(j)):
        export.append([j[i], kt[i], kq[i], eta[i]])
    export = np.array(export)

    np.savetxt('./output/'+filename, export, header='j  kt  kq  eta')

# import all predifined geometry and xfoil computations
def setup():
    rn07 = 9.78e7
    prop_c = PropellerCharacteristics(4.65, 4, 1.1, 0.65)
    water = Fluid(1025, 1.08e-6, 2160, 101325)
    geo = import_geometry("./input/Geometry.txt")

    prop = Propeller(geo, prop_c)

    j = np.linspace(0.4, 1.1, 8)
    
    cl_a = import_cl('./input/')
    return rn07, water, prop, j, cl_a

def q2():
    
    rn07, water, prop, j, cl_a = setup()
    
    # angular and axial speed
    n, v = n_v_from_j_rn(j, rn07, water, prop)
    
    T = np.zeros(len(n))
    Q = np.zeros(len(n))
    
    # Iterate through the blade sections
    for section in range(len(prop.geo.r_R)-1):
        # Lift from hydrodynamic pitch angle
        beta = beta_mean(n, v, prop, section)
        aoa_ = aoa(beta, prop, section)
        cl = cl_from_xfoil(aoa_, cl_a, section)
        v_inf = np.sqrt(v**2 + (prop.geo.r_R[section]*prop.char.d*pi*n)**2)
        
        # Drag coefficient
        cd = cd_ittc(v_inf, prop, water, section)
        
        dr = dr_section(prop, section)
        
        # Drag and lift
        dD = water.density/2.*v_inf**2*cd*prop.geo.c_d[section]*prop.char.d*dr
        dL = water.density/2.*v_inf**2*cl*prop.geo.c_d[section]*prop.char.d*dr
        
        # Thrust and torque
        dT = dL*np.cos(beta) - dD*np.sin(beta)
        dQ = prop.geo.r_R[section]*prop.char.d*(dL*np.sin(beta) + dD*np.cos(beta))/2.
        T += dT
        Q += dQ
    
    # K_T, K_Q and efficiency
    kt = T*prop.char.z /(water.density * n**2 * prop.char.d**4)
    kq = Q*prop.char.z /(water.density * n**2 * prop.char.d**5)
    eta = kt*j/(kq*2*pi)
    
    export('q2.txt', j, kt, kq, eta)

# Lifting line with complete momentum theory
def q3():
    
    # iterations control varialbes
    max_it = 200
    tol = 1e-6
    relax = 0.1
    
    rn07, water, prop, j, cl_a = setup()
    
    # angular and axial speed
    n, v = n_v_from_j_rn(j, rn07, water, prop)
    
    T = np.zeros(len(n))
    Q = np.zeros(len(n))
    
    for i in range(len(j)):
        k = 0
        rel_diff = 1.
        ut = np.zeros(len(prop.geo.r_R)-1)
        ua = np.zeros(len(prop.geo.r_R)-1)
        
        gamma = np.ones(len(prop.geo.r_R)-1)*0.1
        gamma[0] = 0.
        gamma_new = np.zeros(len(prop.geo.r_R)-1)
        cl = np.zeros(len(prop.geo.r_R)-1)
        cd = np.zeros(len(prop.geo.r_R)-1)
        dT = np.zeros(len(prop.geo.r_R)-1)
        dQ = np.zeros(len(prop.geo.r_R)-1)
        
        while k < max_it and rel_diff > tol:
            # Induced velocities
            ut = gamma/(pi*prop.geo.r_R[:-1]*prop.char.d)
            delta = v[i]**2 + ut*(2.*pi*n[i]*prop.geo.r_R[:-1]*prop.char.d-ut)
            ua = -v[i]+np.sqrt(delta)
            
            # Lift from hydrodynamic pitch angle
            for section in range(len(prop.geo.r_R)-1):
                if ut[section] == 0.:
                    ua[section] = 0.1
                beta = np.arctan(ut[section]/ua[section])
                aoa_ = aoa(beta, prop, section)
                cl[section] = cl_from_xfoil(aoa_, cl_a, section)
            v_inf = np.sqrt((v[i] + ua/2.)**2 + (prop.geo.r_R[:-1]*prop.char.d*pi*n[i] - ut/2.)**2)
            # Drag from ittc correlation
            for section in range(len(prop.geo.r_R)-1):
                cd[section] = cd_ittc(v_inf[section], prop, water, section)
            
            # Gamma update, under relaxation to improve convergence
            gamma_new = cl*v_inf*prop.geo.c_d[:-1]*prop.char.d/(2./prop.char.z)
            if k == 0:
                rel_diff = 1.
            else:
                rel_diff = np.linalg.norm(gamma_new - gamma) / np.mean(gamma)
            gamma = gamma*(1-relax) + gamma_new*relax
            
            k += 1
            
            if k % 10 == 0:
                print(k, rel_diff)
        print(k, rel_diff)
        # Computation of cl and cd for the last gamma
        ut = gamma/(pi*prop.geo.r_R[:-1]*prop.char.d)
        delta = v[i]**2 + ut*(2.*pi*n[i]*prop.geo.r_R[:-1]*prop.char.d-ut)
        ua = -v[i]+np.sqrt(delta)
        beta = np.arctan(ut/ua)
        for section in range(len(prop.geo.r_R)-1):
            if ut[section] == 0.:
                ua[section] = 0.1
            aoa_ = aoa(beta[section], prop, section)
            cl[section] = cl_from_xfoil(aoa_, cl_a, section)
        v_inf = np.sqrt((v[i] + ua/2.)**2 + (prop.geo.r_R[:-1]*prop.char.d*pi*n[i] - ut/2.)**2)
        for section in range(len(prop.geo.r_R)-1):
            cd[section] = cd_ittc(v_inf[section], prop, water, section)
        dr = np.array([dr_section(prop, section) for section in range(len(prop.geo.r_R)-1)])
        
        # Drag and lift
        dD = water.density/2.*v_inf**2*cd*prop.geo.c_d[:-1]*prop.char.d*dr
        dL = water.density/2.*v_inf**2*cl*prop.geo.c_d[:-1]*prop.char.d*dr
        
        # Thrust abd torque
        dT[:] = dL*np.cos(beta) - dD*np.sin(beta)
        dQ[:] = prop.geo.r_R[:-1]*prop.char.d*(dL*np.sin(beta) + dD*np.cos(beta))/2.
        T[i] += dT.sum()
        Q[i] += dQ.sum()
        
        print("----------")
    
    # K_T, K_Q and efficiency
    kt = T*prop.char.z /(water.density * n**2 * prop.char.d**4)
    kq = Q*prop.char.z /(water.density * n**2 * prop.char.d**5)
    eta = kt*j/(kq*2*pi)
    
    export('q3.txt', j, kt, kq, eta)

# Lifting line with induction factors
def q4():
    
    # iterations control varialbes
    max_it = 200
    tol = 1e-6
    relax = 0.1
    
    rn07, water, prop, j, cl_a = setup()
    
    # angular and axial speed
    n, v = n_v_from_j_rn(j, rn07, water, prop)
    
    T = np.zeros(len(n))
    Q = np.zeros(len(n))
    
    for i in range(len(j)):
        k = 0
        rel_diff = 1.
        ut = np.zeros(len(prop.geo.r_R)-1)
        ua = np.zeros(len(prop.geo.r_R)-1)
        
        gamma = np.ones(len(prop.geo.r_R)-1)*0.1
        gamma[0] = 0.
        gamma_new = np.zeros(len(prop.geo.r_R)-1)
        cl = np.zeros(len(prop.geo.r_R)-1)
        cd = np.zeros(len(prop.geo.r_R)-1)
        dT = np.zeros(len(prop.geo.r_R)-1)
        dQ = np.zeros(len(prop.geo.r_R)-1)
        
        while k < max_it and rel_diff > tol:
            # Induced velocities
            ut = gamma/(pi*prop.geo.r_R[:-1]*prop.char.d)
            delta = v[i]**2 + ut*(2.*pi*n[i]*prop.geo.r_R[:-1]*prop.char.d-ut)
            ua = -v[i]+np.sqrt(delta)
            
            # Lift from hydrodynamic pitch angle
            for section in range(len(prop.geo.r_R)-1):
                if ut[section] == 0.:
                    ua[section] = 0.1
                beta = np.arctan(ut[section]/ua[section])
                aoa_ = aoa(beta, prop, section)
                cl[section] = cl_from_xfoil(aoa_, cl_a, section)
            v_inf = np.sqrt((v[i] + ua/2.)**2 + (prop.geo.r_R[:-1]*prop.char.d*pi*n[i] - ut/2.)**2)
            # Drag from ittc correlation
            for section in range(len(prop.geo.r_R)-1):
                cd[section] = cd_ittc(v_inf[section], prop, water, section)
            
            # Gamma update, under relaxation to improve convergence
            gamma_new = cl*v_inf*prop.geo.c_d[:-1]*prop.char.d/(2./prop.char.z)
            if k == 0:
                rel_diff = 1.
            else:
                rel_diff = np.linalg.norm(gamma_new - gamma) / np.mean(gamma)
            gamma = gamma*(1-relax) + gamma_new*relax
            
            k += 1
            
            if k % 10 == 0:
                print(k, rel_diff)
        print(k, rel_diff)
        
        # Computation of cl and cd for the last gamma
        ut = gamma/(pi*prop.geo.r_R[:-1]*prop.char.d)
        delta = v[i]**2 + ut*(2.*pi*n[i]*prop.geo.r_R[:-1]*prop.char.d-ut)
        ua = -v[i]+np.sqrt(delta)
        beta = np.arctan(ut/ua)
        for section in range(len(prop.geo.r_R)-1):
            if ut[section] == 0.:
                ua[section] = 0.1
            aoa_ = aoa(beta[section], prop, section)
            cl[section] = cl_from_xfoil(aoa_, cl_a, section)
        v_inf = np.sqrt((v[i] + ua/2.)**2 + (prop.geo.r_R[:-1]*prop.char.d*pi*n[i] - ut/2.)**2)
        for section in range(len(prop.geo.r_R)-1):
            cd[section] = cd_ittc(v_inf[section], prop, water, section)
        dr = np.array([dr_section(prop, section) for section in range(len(prop.geo.r_R)-1)])
        
        # Drag and lift
        dD = water.density/2.*v_inf**2*cd*prop.geo.c_d[:-1]*prop.char.d*dr
        dL = water.density/2.*v_inf**2*cl*prop.geo.c_d[:-1]*prop.char.d*dr
        
        # Thrust abd torque
        dT[:] = dL*np.cos(beta) - dD*np.sin(beta)
        dQ[:] = prop.geo.r_R[:-1]*prop.char.d*(dL*np.sin(beta) + dD*np.cos(beta))/2.
        T[i] += dT.sum()
        Q[i] += dQ.sum()
        
        print("----------")
    
    # K_T, K_Q and efficiency
    kt = T*prop.char.z /(water.density * n**2 * prop.char.d**4)
    kq = Q*prop.char.z /(water.density * n**2 * prop.char.d**5)
    eta = kt*j/(kq*2*pi)
    
    export('q4.txt', j, kt, kq, eta)
    
print("***************************")
print("q2")
print("***************************")
q2()
print()
print("***************************")
print("q3")
print("***************************")
print()
q3()
print()
print("***************************")
print("q4")
print("***************************")
print()
q4()