import numpy as np
import matplotlib.pyplot as plt
from InductionFactors import *
from SingularIntegration import *
from numpy import pi
from utils import *
from scipy import interpolate

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
    cl_cubic = interpolate.CubicSpline(cl_a[section, :, 0], cl_a[section, :, 1])
    #cl = np.interp(rad_to_deg(a), cl_a[section, :, 0], cl_a[section, :, 1])
    cl = cl_cubic(rad_to_deg(a))
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
                beta = np.arctan((v[i] + ua[section]/2.)/(prop.geo.r_R[section]*prop.char.d*pi*n[i] - ut[section]/2.))
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
        beta = np.arctan((v[i] + ua/2.)/(prop.geo.r_R[:-1]*prop.char.d*pi*n[i] - ut/2.))
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

def sinspace(start, end, num=50):
    # Create a sinusoidal distribution over the interval
    t = np.linspace(0, np.pi, num)  # Create a sine-shaped spacing
    s = (1 - np.cos(t)) / 2         # Map cosine to [0, 1] for smoother spacing
    return start + (end - start) * s

# Lifting line with induction factors
def q4():
    
    # iterations control varialbes
    max_it = 3000
    tol = 1e-6
    relax = 0.01
    
    # Span-wise discretisation (can be higher than the number of provided foil sections)
    n_span = 40
    
    rn07, water, prop, j, cl_a = setup()
    
    # angular and axial speed
    n, v = n_v_from_j_rn(j, rn07, water, prop)
    
    T = np.zeros(len(n))
    Q = np.zeros(len(n))
    
    for i in range(len(j)):
        k = 0
        rel_diff = 1.
        ut_2 = np.zeros(n_span)
        ua_2 = np.zeros(n_span)
        
        # _2 indicates that the discretisation is the one controlled by n_span and not the provided geometry
        gamma_2 = np.ones(n_span)*0.1
        #gamma_2 = np.array([-((float(l) - float(n_span)/2.)/float(n_span))**2 + 1. for l in range(n_span)])
        gamma_2[0] = 0.
        gamma_2[-1] = 0.
        gamma_new_2 = np.zeros(n_span)
        cl = np.zeros(len(prop.geo.r_R))    # To ensure proper gamma at the tip, see later
        cl_2 = np.zeros(n_span)
        cd = np.zeros(len(prop.geo.r_R)-1)
        dT_2 = np.zeros(n_span)
        dQ_2 = np.zeros(n_span)
        beta = np.zeros(len(prop.geo.r_R)-1)
        beta_2 = np.zeros(n_span)
        r_2 = np.linspace(prop.geo.r_R[0]*prop.char.d/2., prop.geo.r_R[-1]*prop.char.d/2., n_span)
        #r_2 = sinspace(prop.geo.r_R[0]*prop.char.d/2., prop.geo.r_R[-1]*prop.char.d/2., n_span)
        beta_2[:] = np.arctan(v[i]/(2*pi*r_2*n[i]))
        c_2 = np.interp(r_2, prop.geo.r_R*prop.char.d/2., prop.geo.c_d*prop.char.d)
        
        plt.plot(r_2, gamma_2, 'x-', label=k)
        
        while k < max_it and rel_diff > tol:
            # Lift from hydrodynamic pitch angle
            for section in range(len(prop.geo.r_R)-1):
                aoa_ = aoa(beta[section], prop, section)
                cl[section] = cl_from_xfoil(aoa_, cl_a, section)
            #cl[0] = 0.         #debug
            cl[-1] = 0.     # No foil at the tip
            #print('cl ', cl)
            cl_2 = np.interp(r_2, prop.geo.r_R[:]*prop.char.d/2., cl)
            v_inf_2 = np.sqrt((v[i] + ua_2/2.)**2 + (2*r_2*pi*n[i] - ut_2/2.)**2)
            
            #plt.plot(r_2, cl_2, 'x-', label=k+1)
            
            # Gamma update, under relaxation to improve convergence
            gamma_new_2 = prop.char.z/2. * cl_2*v_inf_2*c_2
            if k == 0:
                rel_diff = 1.
            else:
                old_rel_diff = rel_diff
                rel_diff = np.linalg.norm(gamma_new_2 - gamma_2) / np.mean(gamma_2)
                # if abs(old_rel_diff-rel_diff)/rel_diff < 1e-4:
                #     relax *= 2.if abs(old_rel_diff-rel_diff)/rel_diff < 1e-4:
                #     relax *= 2.
            gamma_2 = gamma_2*(1-relax) + gamma_new_2*relax
            
            gamma_2[0] = 0.
            gamma_2[-1] = 0.
            
            # if rel_diff > 15.:
            #     for l in range(1, n_span-1):
            #         gamma_2[l] = 0.5*(gamma_2[l-1] + gamma_2[l+1])
            
            #plt.plot(r_2, gamma_2, 'x-', label=k+1)
            # Derivative of Gamma according to r
            dg_dr_2 = np.zeros(n_span)
            dg_dr_2[0] = (gamma_2[1] - gamma_2[0])/(r_2[1] - r_2[0])#/prop.char.z
            dg_dr_2[-1] = (gamma_2[-1] - gamma_2[-2])/(r_2[-1] - r_2[-2])#/prop.char.z
            for l in range(1, n_span-1):
                dg_dr_2[l] = (gamma_2[l+1] - gamma_2[l-1])/(r_2[l+1] - r_2[l-1])#/prop.char.z
            
            plt.plot(r_2, dg_dr_2, 'x-', label=k+1)
            
            ia_2 = np.zeros(n_span)
            it_2 = np.zeros(n_span)
            for l in range(1,n_span-1):
                # Induction factors
                for m in range(n_span):
                    ia_2[m], it_2[m] = inductionFactors(r_2[m], r_2[l], beta_2[m], prop.char.z)
                ua_2[l] = singularIntegration(r_2, dg_dr_2*ia_2, r_2[l])/(2.*pi)
                ut_2[l] = singularIntegration(r_2, dg_dr_2*it_2, r_2[l])/(2.*pi)
            #print('NaN in ut ua', np.sum(np.isnan(ut_2)), np.sum(np.isnan(ua_2)))
            if np.sum(np.isnan(ut_2)) > 0:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.show()
                print('EXIT - NaN Values')
                exit()
            ua_2[0] = 0.
            ua_2[-1] = 0.
            ut_2[0] = 0.
            ut_2[-1] = 0.
            
            # Hydrodynamic pitch angle
            #print('ut ua ', ut_2, ua_2)
            beta_2 = np.arctan((v[i] + ua_2/2.)/(2*r_2*pi*n[i] - ut_2/2.))
            beta = np.interp(prop.geo.r_R[:-1]*prop.char.d/2., r_2, beta_2)
            k += 1
            #print('beta ', beta_2)
            if k % 1 == 0:
                print(k, rel_diff, np.sum(gamma_2), relax)
            #print('ooooooooooo')
        
        print(k, rel_diff)
        
        plt.plot(r_2, gamma_2, label=i)
        # Derivative of Gamma according to r
        dg_dr_2 = np.zeros(n_span)
        dg_dr_2[0] = (gamma_2[1] - gamma_2[0])/(r_2[1] - r_2[0])
        dg_dr_2[-1] = (gamma_2[-1] - gamma_2[-2])/(r_2[-1] - r_2[-2])
        for l in range(1, n_span-1):
            dg_dr_2[l] = (gamma_2[l+1] - gamma_2[l-1])/(r_2[l+1] - r_2[l-1])
        
        ia_2 = np.zeros(n_span)
        it_2 = np.zeros(n_span)
        
        for l in range(n_span-1):
            # Induction factors
            for m in range(n_span):
                ia_2[m], it_2[m] = inductionFactors(r_2[m], r_2[l], beta_2[l], prop.char.z)
            ua_2[l] = 0.5*singularIntegration(r_2, dg_dr_2*ia_2, r_2[l])*v[i]
            ut_2[l] = 0.5*singularIntegration(r_2, dg_dr_2*it_2, r_2[l])*v[i]
        ua_2[0] = 0.1
        ua_2[-1] = 0.1
        ut_2[0] = 0.
        ut_2[-1] = 0.
        
        # Hydrodynamic pitch angle
        beta_2 = np.arctan((v[i] + ua_2/2.)/(2*r_2*pi*n[i] - ut_2/2.))
        beta = np.interp(prop.geo.r_R[:-1]*prop.char.d/2., r_2, beta_2)
        ut = np.interp(prop.geo.r_R[:-1]*prop.char.d/2., r_2, ut_2)
        ua = np.interp(prop.geo.r_R[:-1]*prop.char.d/2., r_2, ut_2)
        
        for section in range(len(prop.geo.r_R)-1):
            aoa_ = aoa(beta[section], prop, section)
            cl[section] = cl_from_xfoil(aoa_, cl_a, section)
        cl[-1] = 0.
        v_inf = np.sqrt((v[i] + ua/2.)**2 + (prop.geo.r_R[:-1]*prop.char.d*pi*n[i] - ut/2.)**2)
        cl_2 = np.interp(r_2, prop.geo.r_R[:]*prop.char.d/2., cl)
        v_inf_2 = np.sqrt((v[i] + ua_2/2.)**2 + (2*r_2*pi*n[i] - ut_2/2.)**2)
        for section in range(len(prop.geo.r_R)-1):
            cd[section] = cd_ittc(v_inf[section], prop, water, section)
        cd_2 = np.interp(r_2, prop.geo.r_R[:-1]*prop.char.d/2., cd)
        dr_2 = np.zeros(n_span)
        dr_2[0] = (r_2[1] - r_2[0])/2.
        dr_2[-1] = (r_2[-1] - r_2[-2])/2.
        for l in range(1, n_span-1):
            dr_2 = (r_2[l+1] - r_2[l-1])/2.
        
        c = np.interp(r_2, prop.geo.r_R*prop.char.d/2., prop.geo.c_d*prop.char.d)
        # Drag and lift
        dD_2 = water.density/2.*v_inf_2**2*cd_2*c*dr_2
        dL_2 = water.density/2.*v_inf_2**2*cl_2*c*dr_2
        
        # Thrust abd torque
        dT_2[:] = dL_2*np.cos(beta_2) - dD_2*np.sin(beta_2)
        dQ_2[:] = r_2*(dL_2*np.sin(beta_2) + dD_2*np.cos(beta_2))/2.
        T[i] += dT_2.sum()
        Q[i] += dQ_2.sum()
        
        print(T*prop.char.z /(water.density * n**2 * prop.char.d**4))
        exit()
        
        print("----------")
    
    plt.show()
    # K_T, K_Q and efficiency
    kt = T*prop.char.z /(water.density * n**2 * prop.char.d**4)
    kq = Q*prop.char.z /(water.density * n**2 * prop.char.d**5)
    eta = kt*j/(kq*2*pi)
    
    export('q4.txt', j, kt, kq, eta)
    
# print("***************************")
# print("q2")
# print("***************************")
# q2()
# print()
# print("***************************")
# print("q3")
# print("***************************")
# print()
# q3()
# print()
print("***************************")
print("q4")
print("***************************")
print()
q4()