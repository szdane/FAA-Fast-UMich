# import gurobipy as gpfrom 
# gurobipy 
# import GRB
# import math 

# def f(u):  return math.atan(u)

# m = gp.Model()
# lbz = -2
# ubz =  2
# z = m.addVar(lb=lbz, ub=ubz, vtype=GRB.CONTINUOUS, name="z")
# y = m.addVar(lb=-math.pi/2, ub=math.pi/2, vtype=GRB.CONTINUOUS, name="y")# Compute piecewise-linear arctan function for z
# npts = 101
# ptu = []
# ptf = []
# for i in range(npts):    
#     ptu.append(lbz + (ubz - lbz) * i / (npts - 1))    
#     ptf.append(f(ptu[i]))

# # Add constraint y = arctan(z) as piecewise-linear approximation

# m.addGenConstrPWL(z,y,ptu,ptf,"PWLarctan")


import numpy as np
import math

def f(u):  return math.atan(u)
lbx = -2
ubx =  2    
npts = 101
x_pts = []
y_pts = []
for i in range(npts):    
    x_pts.append(lbx + (ubx - lbx) * i / (npts - 1))    
    y_pts.append(f(x_pts[i]))


approx = np.interp(1, x_pts, y_pts)

real = np.arctan(1)

print("real", real, "approx", approx)

gamma_min = -np.pi / 2
gamma_max = np.pi / 2
num_points = 100  # Number of points for the piecewise-linear approximation

# Generate points and their cosine values
gamma_points = np.linspace(gamma_min, gamma_max, num_points)
cos_values = np.cos(gamma_points)

approx_cos = np.interp(1, gamma_points, cos_values)
real_cos = np.cos(1)

print("real cos", real_cos, "approx cos", approx_cos)