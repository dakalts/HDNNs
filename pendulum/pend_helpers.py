#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dimitrios Kaltsas
Hamilton-Dirac-PINNs

Helper functions for planar pendulum HDNN
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
   
   
# right hand side of planar pendulum odes  
def f(t, u, m, g):
    x, y, px, py = u    
    M = -(x*px+y*py)/(m*(x**2+y**2))
    L = -(px**2/m + py**2/m -m*g*y)/(x**2+y**2)
    derivs = [px/m+M*x, py/m+M*y, L*x-M*px, L*y-M*py-m*g]
    return derivs

# RK45 (Scipy) 
def RK45(N, x0, y0, px0, py0, t0, t_max, m, g):
    t = np.linspace(t0, t_max, N+1)
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    t_span = (t[0],t[-1])
    sol = solve_ivp(f, t_span, u0, args=(m,g), method='RK45', t_eval=t)
    xP = sol.y[0,:]    
    yP  = sol.y[1,:]
    pxP = sol.y[2,:]   
    pyP = sol.y[3,:]
    E_rk45 = energy( xP, yP, pxP, pyP, m,g)
    return E_rk45, xP, yP, pxP, pyP, t

# Scipy Solver   
def LSODA(N,t, x0, y0, px0, py0, m, g):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(f, u0, t, args=(m,g), tfirst=True)
    xP = solPend[:,0]    
    yP  = solPend[:,1]
    pxP = solPend[:,2]  
    pyP = solPend[:,3]
    return xP,yP, pxP, pyP

# Energy of the system
def energy(x, y, px, py, m, g):    
    Nx=len(x) 
    x=x.reshape(Nx)      
    y=y.reshape(Nx)
    px=px.reshape(Nx)    
    py=py.reshape(Nx)
    E = (px**2 + py**2)/(2*m) + m*g*(1+y)
    E = E.reshape(Nx)
    return E



def saveData(path, t, x, y, px,py, E):
    np.savetxt(path+"t.txt",t)
    np.savetxt(path+"x.txt",x)
    np.savetxt(path+"y.txt",y)
    np.savetxt(path+"px.txt",px)
    np.savetxt(path+"py.txt",py)
    np.savetxt(path+"E.txt",E)

    
    