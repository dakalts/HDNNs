#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dimitrios Kaltsas
Hamilton-Dirac-PINNs

Helper functions for the GC motion

"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
   


# Right hand side of gc odes 
def f(t, u , mu, epsilon, kappa):
    x, y, px, py = u      # unpack current values of u
    B = 1 + epsilon*(x**2/kappa+y**2)
    Wx = 2*mu*epsilon*x/kappa
    Wy = mu*epsilon*2*y
    derivs = [-Wy/B, Wx/B, -0.5*Wx+epsilon*x*y*Wy/(kappa*B), -0.5*Wy+epsilon*x*y*Wx/(B)]     # list of dy/dt=f functions
    return derivs

# RK45 (Scipy) 
def RK45(N, x0, y0, px0, py0,t0,t_max, mu, epsilon, kappa):
    t = np.linspace(t0, t_max, N+1)
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    t_span = (t[0],t[-1])
    sol = solve_ivp(f, t_span, u0, args=(mu,epsilon,kappa), method='RK45',t_eval=t)
    xP = sol.y[0,:]    
    yP  = sol.y[1,:]
    pxP = sol.y[2,:]   
    pyP = sol.y[3,:]
    E_rk45 = energy( xP, yP, pxP, pyP, mu , epsilon, kappa)
    return E_rk45,xP,yP, pxP, pyP, t

# Scipy LSODA Solver   
def LSODA(N, t, x0, y0, px0, py0, mu, epsilon, kappa):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    sol = odeint(f, u0, t, args=(mu,epsilon,kappa),tfirst=True)
    xP = sol[:,0]    
    yP  = sol[:,1]
    pxP = sol[:,2]   
    pyP = sol[:,3]
    return xP,yP, pxP, pyP

# Energy of the guiding center system
def energy(x, y, px, py, mu, epsilon, kappa):    
    Nx=len(x) 
    x=x.reshape(Nx)      
    y=y.reshape(Nx)
    px=px.reshape(Nx)    
    py=py.reshape(Nx)
    Ax =-0.5*(y+epsilon*(x**2*y/kappa+y**3/3))
    Ay = 0.5*(x+epsilon*(x**3/(3*kappa)+y**2*x))
    B=(1+epsilon*(x**2/kappa+y**2))
    W = mu*B
    Wx = 2*mu*epsilon*x/kappa
    Wy = mu*epsilon*2*y
    E = W + 0*(- (px-Ax)*Wy/B +(py-Ay)*Wx/B )
    E = E.reshape(Nx)
    return E


def saveData(path, t, x, y, px,py, E):
    np.savetxt(path+"t.txt",t)
    np.savetxt(path+"x.txt",x)
    np.savetxt(path+"y.txt",y)
    np.savetxt(path+"px.txt",px)
    np.savetxt(path+"py.txt",py)
    np.savetxt(path+"E.txt",E)

    
    