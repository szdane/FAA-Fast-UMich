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