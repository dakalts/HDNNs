#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dimitrios Kaltsas

Code for the paper:
"Constraint Hamiltonian systems and physics-informed neural networks: Hamilton-Dirac neural networks"
by D. A. Kaltsas, Physical Review E 111, 025301 (2025).

Example 2: HDNN for the elliptically restricted harmonic oscillator (ERHO)

First version: 25/12/2023
Last updated: 27/01/2025
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from os import path
import sys
from IPython import get_ipython


# Set up plotting and clear cache
get_ipython().run_line_magic('matplotlib', 'qt')
plt.close('all')
torch.cuda.empty_cache()  # Release unoccupied cached memory
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print(f"Using device: {device}")

# Custom activation function: Sinegmoid
class Sinegmoid(torch.nn.Module):
    def __init__(self,delta1,delta2):
        super().__init__()
        self.delta1 = delta1
        self.delta2 = delta2
    def forward(self, x):
        return torch.sin(self.delta1*x)*torch.sigmoid(self.delta2*x)


# Auto-differentiation for derivatives
def dfx(x,f):
    
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

#rhs of constrained oscillator
def f(t, u, alpha, beta, epsilon):
    x, y, px, py = u
    mu = - (x*px+y*py/(1-epsilon**2))/(x**2+y**2/(1-epsilon**2)**2)
    lambd= -(px**2+py**2/(1-epsilon**2)-alpha*(x**2)-beta*(y**2)/(1-epsilon**2))/(x**2+y**2/(1-epsilon**2)**2)
    F1 =  px + mu*x 
    F2 = py + mu*y
    F3 = - alpha*(x) - mu*px + lambd*x
    F4 = - beta*(y) - mu*py + lambd*y
    
    derivs = [F1,F2,F3,F4]
    return derivs

# rhs of unrestricted oscillator
def F_un(t, u, alpha, beta, epsilon):
    x, y, px, py = u 
    F1 =  px 
    F2 = py 
    F3 = - alpha*(x) 
    F4 = - beta*(y) 
    derivs = [F1,F2,F3,F4]
    return derivs

# Perturb evaluation points stochastically
def perturbPoints(grid,t0,tf,sig=0.5):
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[2] = torch.ones(1,1)*(-1)
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    t.requires_grad = False
    return t

# Parametric solutions for the system
def parametricSolutions(t, nn, P0, epsilon,gamma):
    t0, x0, y0, px0, py0= P0[0],P0[1],P0[2],P0[3],P0[4]
    N1, N2, N3, N4 = nn(t,epsilon)
    dt =t-t0
    f = (1-torch.exp(-gamma*dt))
    x_hat  = x0  + f*N1
    y_hat  = y0  + f*N2
    px_hat = px0 + f*N3
    py_hat = py0 + f*N4

    return x_hat, y_hat, px_hat, py_hat


def HamDirEqs_Loss(t,x,y,px,py, alpha, beta, epsilon):
    # Define the loss function by Hamilton Eqs., write explicitely the Hamilton-Dirac Equations
    xd, yd, pxd, pyd =\
        dfx(t,x),dfx(t,y),dfx(t,px),dfx(t,py)
    F1, F2, F3, F4 = f(t,[x,y,px,py],alpha,beta,epsilon)
    fx  = xd - F1
    fy  = yd - F2
    fpx = pxd -F3
    fpy = pyd -F4
    Lx  = (fx.pow(2)).mean()  
    Ly  = (fy.pow(2)).mean()
    Lpx = (fpx.pow(2)).mean()
    Lpy = (fpy.pow(2)).mean()

    L = Lx + Ly + Lpx + Lpy
    return L



def hamiltonian(x,  y, px, py, x0, y0, alpha, beta, epsilon):
    #returns the hamiltonian ham for Kinetic (K)  and Potential (V) Energies
    l = np.sqrt(x0**2 + y0**2/(1-epsilon**2))
    
    V = 0.5*alpha*(x**2) + 0.5*beta*(y**2)
    K = (px**2+py**2)/2 

    
    ham = K + V 
    return ham

              
def prim_constr_loss(x,y,x0,y0,epsilon):
    ls = x0**2+y0**2/(1-epsilon**2)
    Phi = x**2+y**2/(1-epsilon**2)-ls
    Lphi=Phi.pow(2).mean()
    return Lphi


def second_constr_loss(x, y, px, py,x0,y0,px0,py0,epsilon):
    Psi = x*px+y*py/(1-epsilon**2)-(x0*px0+y0*py0/(1-epsilon**2))
    Lpsi=Psi.pow(2).mean()
    return Lpsi



# NETWORK 



class HDNN(torch.nn.Module):
    def __init__(self, inpn, outn, hidn, D_hid, actF):
        super(HDNN,self).__init__()

        # Define the Activation
        self.actF = actF
        # define layers
        self.Lin_1   = torch.nn.Linear(inpn, D_hid)
        self.Lin_hid   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, outn)
        
    '''
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    '''

    def forward(self,t,epsilon):
        # layer 1
        ones = torch.ones_like(t)
        inputs = torch.cat([t, ones*epsilon], axis=1)
  
    
        l =self.Lin_1(inputs)
        h = self.actF(l)
        for i in range(hidn-1):
            l = self.Lin_hid(h)    
            h = self.actF(l)

        # output layer
        r = self.Lin_out(h)
        x=[(r[:,i]).reshape(-1,1) for i in range(outn)]
        return x



# Weight initialization
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        y = m.in_features
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        m.bias.data.fill_(0)

# Train the HDNN
def train_HDNN(
    P0, tf, inpn, outn, hidn, neurons, epochs, n_train, lr, 
    loadWeights, minLoss, PATH="models/model_HDNN_erho"):
    """
    Train the HDNN for ERHO system
    
    Parameters:
    - P0: Initial conditions.
    - tf: Final time.
    - inpn, outn: Input and output sizes.
    - hidn: Number of hidden layers.
    - neurons: Number of neurons per hidden layer.
    - epochs: Number of training epochs.
    - n_train: Number of training samples.
    - lr: Learning rate.
    - loadWeights: Boolean flag to load pre-trained weights.
    - minLoss: Minimum loss threshold for early stopping.
    - PATH: Path for saving/loading model weights.
    
    Returns:
    - Trained model and loss histories.
    """

    # Initialize the HDNN model
    fc0 = HDNN(inpn, outn, hidn, neurons, actF).to(device)
    fc0.apply(weights_init_normal)  # Initialize weights with normal distribution
    fc1 = copy.deepcopy(fc0)  # Deep copy for storing the best model

    # Optimizer (includes trainable parameters gamma, delta1, delta2)
    optimizer = optim.Adam(list(fc0.parameters()) + [gamma, delta1, delta2], lr=lr)

    # Loss limit for best model tracking
    Llim = 1  

    # Unpack initial conditions
    t0, x0, y0, px0, py0, alpha, beta, epsilon1, epsilon2 = P0

    # Generate time grid
    grid = torch.linspace(t0, tf, n_train).reshape(-1, 1)

    # Load pre-trained weights if specified
    if loadWeights and path.exists(PATH):
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        fc0.train()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.1, total_steps=epochs)

    # Initial energy computation
    TeP0 = time.time()
    t_num = np.linspace(t0, tf, n_train)
    x_num, y_num, px_num, py_num = LSODA(N, t_num, x0, y0, px0, py0, alpha, beta, (epsilon1 + epsilon2) / 2)

    # Training loop
    for tt in range(epochs):                
        
        # Perturb evaluation points while keeping t[0] = t0
        t = perturbPoints(grid, t0, tf, sig=0.3 * tf).to(device)
        t = t[torch.randperm(t.shape[0])].to(device)  # Shuffle time points
        t.requires_grad = True
        
        # Update epsilon values every 100 epochs
        if tt % 100 == 0:
            epsilon = torch.from_numpy(np.random.uniform(epsilon1, epsilon2, size=(N, 1))).float().to(device)
            ham0 = hamiltonian(x0, y0, px0, py0, x0, y0, alpha, beta, epsilon.cpu().detach().numpy()) 
        
        # Compute parametric solutions for training
        x_, y_, px_, py_ = parametricSolutions(t, fc0, P0, (epsilon1 + epsilon2) / 2, gamma)
        x, y, px, py = parametricSolutions(t, fc0, P0, epsilon, gamma)
        
        # Compute losses
        L_eq = HamDirEqs_Loss(t, x, y, px, py, alpha, beta, epsilon)  # Hamilton equations loss
        ham = hamiltonian(x, y, px, py, x0, y0, alpha, beta, epsilon.cpu().detach().numpy())
        L_en = 0.5 * ((ham - ham0).pow(2)).mean()  # Energy regularization loss
        L_constr = (
            prim_constr_loss(x, y, x0, y0, epsilon) + 
            second_constr_loss(x, y, px, py, x0, y0, px0, py0, epsilon)
        )  # Constraint loss
        
        # Total loss
        Ltot = w_eq * L_eq + w_en * L_en + w_constr * L_constr
        
        # Backpropagation and optimization
        Ltot.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        # Record loss history
        Loss_history.append(Ltot.item())     
        Loss_eq_history.append(L_eq.item())
        Loss_en_history.append(L_en.item())
        Loss_constr_history.append(L_constr.item())

        # Keep the best model (lowest loss)
        if tt > 0.8 * epochs and Ltot < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = Ltot 

        # Early stopping if loss is below threshold
        if Ltot < minLoss:
            fc1 = copy.deepcopy(fc0)
            print("Reached minimum requested loss")
            break

        # Periodic logging and visualization
        if tt % 50 == 0:    
            print(f" epoch: {tt}, L_eq = {L_eq:.5f}, L_en = {L_en:.5f}, L_constr = {L_constr:.5f},\
                  lr={lr:.5f}, t_max={t_max:.3f}, N={n_train}, gamma = {gamma:.3f},\
                      delta1 = {delta1:.3f}, delta2 = {delta2:.3f}\
                      --------------------------------------------------------------------------")
            
            plt.clf()
            plt.scatter(px_[:80].cpu().detach().numpy(), py_[:80].cpu().detach().numpy(), s=3, label="HDNN")
            plt.scatter(px_num[:80], py_num[:80], s=3, label="Exact")
            plt.xlabel(r"$p_x$", fontsize=14)
            plt.ylabel(r"$p_y$", fontsize=14, rotation=0)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.gca().set_aspect("equal")
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=14)
            plt.pause(0.1)
            plt.show()

    # Compute runtime
    runTime = time.time() - TeP0  
    
    # Save the trained model
    torch.save({
        "epoch": tt,
        "model_state_dict": fc1.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": Ltot,
    }, PATH)

    return fc1, Loss_history, Loss_eq_history, Loss_en_history, Loss_constr_history, runTime

    

def loadModel(PATH="models/model_HDNN_erho"):
    if path.exists(PATH):
        fc0 = HDNN(inpn,outn,hidn,neurons,actF)
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train(); # or model.eval
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()
    return fc0.to(device)


from scipy.integrate import odeint, solve_ivp 

# RK45 Solver (Scipy) 
def RK45(N, x0, y0, px0, py0, t0, t_max, alpha,beta,epsilon):
    t = np.linspace(t0, t_max, N+1)
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    t_span = (t[0],t[-1])
    sol = solve_ivp(f, t_span, u0, args=(alpha,beta,epsilon), method='RK45', t_eval=t)
    xP = sol.y[0,:]    
    yP  = sol.y[1,:]
    pxP = sol.y[2,:]   
    pyP = sol.y[3,:]
    E_rk45 = energy( xP, yP, pxP, pyP,  alpha,beta,epsilon)
    return E_rk45, xP, yP, pxP, pyP, t

# Scipy Solver LSODA 
def LSODA(N,t, x0, y0, px0, py0, alpha,beta, epsilon):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solerho = odeint(f, u0, t, args=(alpha,beta,epsilon), tfirst=True)
    xP = solerho[:,0]    
    yP  = solerho[:,1]
    pxP = solerho[:,2]  
    pyP = solerho[:,3]
    return xP,yP, pxP, pyP

def LSODA_un(N,t, x0, y0, px0, py0, alpha,beta, epsilon):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solerho = odeint(F_un, u0, t, args=(alpha,beta,epsilon), tfirst=True)
    xP = solerho[:,0]    
    yP  = solerho[:,1]
    pxP = solerho[:,2]  
    pyP = solerho[:,3]
    return xP,yP, pxP, pyP

# Energy of the system
def energy(x, y, px, py, alpha, beta,epsilon):    
    Nx=len(x) 
    x=x.reshape(Nx)      
    y=y.reshape(Nx)
    px=px.reshape(Nx)    
    py=py.reshape(Nx)
    V = 0.5*alpha*(x**2) + 0.5*beta*(y**2)
    K = (px**2+py**2)/2 
    E = K+V
    E = E.reshape(Nx)
    return E

    

def loss_plots(loss,loss_eq,loss_erg,loss_constr,w_constr):
    """
    Plots various loss metrics with logarithmic scaling.
    
    variables:
    loss (array-like): Total loss values over epochs.
    loss_eq (array-like): Equation loss values.
    loss_erg (array-like): Energy loss values.
    loss_constr (array-like): Constraint loss values.
    w_constr (float): Constraint weight used for labeling the saved figure.
    """
    groupsize = 100  # Number of epochs per averaging group
    # Group losses into chunks of 'groupsize' for averaging
    def group_and_calculate_stats(data):
        grouped = np.array([data[x:x+groupsize] for x in range(0, len(data), groupsize)])
        return grouped.mean(axis=1), grouped.std(axis=1)
    
    mean_loss, std_loss = group_and_calculate_stats(loss)
    mean_loss_eq, std_loss_eq = group_and_calculate_stats(loss_eq)
    mean_loss_erg, std_loss_erg = group_and_calculate_stats(loss_erg)
    mean_loss_constr, std_loss_constr = group_and_calculate_stats(loss_constr)
    
    xm = np.arange(len(mean_loss))  # X-axis values
    
    plt.figure()
    # Plot mean losses with different styles
    plt.loglog(mean_loss,'b',alpha=0.6, linewidth=2.0, label='Total loss')
    plt.loglog(mean_loss_eq,'c--',alpha=0.6, linewidth=2.0, label='Equation loss')
    plt.loglog(mean_loss_erg,'r:',alpha=0.6, linewidth=2.0, label='Energy loss')
    plt.loglog(mean_loss_constr, 'g-.', alpha=0.6, linewidth=2.0, label='Constraint loss')
    
    # Fill between mean ± std for visualization of variance
    plt.fill_between(xm,mean_loss+std_loss, mean_loss-std_loss, facecolor='blue', alpha=0.35)
    plt.fill_between(xm,mean_loss_eq+std_loss_eq, mean_loss_eq-std_loss_eq, facecolor='cyan', alpha=0.35)
    plt.fill_between(xm,mean_loss_erg+std_loss_erg, mean_loss_erg-std_loss_erg, facecolor='red', alpha=0.35)
    plt.fill_between(xm,mean_loss_constr+std_loss_constr, mean_loss_constr-std_loss_constr, facecolor='green', alpha=0.35)
    
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.tight_layout()
    plt.ylabel('Loss',fontsize=22,rotation=90)
    plt.xlabel(r'epochs $(x10^2)$',fontsize=22,rotation=0)    
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    
    # Save the figure
    plt.savefig(f'figs/losses_pend_hd_{w_constr}.pdf', format="pdf", bbox_inches="tight")
    plt.show()
  
    
def load_data(epsilon0,w_constr):
    """
    Loads model and loss data from files, then generates loss plots.
    
    Parameters:
    epsilon0 (float): Mean eccentricity value.
    w_constr (float): Constraint weight used in filenames.
    """
    model = loadModel()
    loss = np.loadtxt('data/loss.txt')
    loss_eq = np.loadtxt('data/loss_eq.txt')
    loss_erg = np.loadtxt('data/loss_erg.txt')
    loss_constr = np.loadtxt('data/loss_constr.txt')
    from pred_solutions import pred_solutions
    pred_solutions(epsilon0,P0,N,t_max,model,device,alpha,beta,gamma,w_constr)
    loss_plots(loss,loss_eq,loss_erg,loss_constr,w_constr)

if __name__ == "__main__":
    
    # training parameter range 
    epsilon1 = 0.1
    epsilon2 = 0.3
    
    # Set the initial state    
    t0, x0, y0, px0, py0, alpha , beta =  0, 0.5, 0.5, 0.2, 0.0,  0.1, 0.4
    P0 = [t0, x0, y0, px0, py0, alpha , beta, epsilon1, epsilon2]
    
    # Define parameters
    N=500
    Tmax= 50
    inpn = 2
    outn = 4
    hidn = 4
    neurons = 160
    
    # trainable parameters
    gamma = nn.Parameter(torch.tensor(1.0),requires_grad=True)
    delta1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
    delta2= nn.Parameter(torch.tensor(1.0),requires_grad=True)
    
    # activation function
    actF= Sinegmoid(delta1,delta2) 
    
    # training hyperparameters
    epochs = 10000
    lr = 5e-3
    E0 = 0.5*(px0**2+py0**2) + 0.5*alpha*(x0**2)+0.5*beta*(y0**2) #Initial Energy
    w_eq = 2.0     # weight of equation residual term
    w_en = 8.0     # weight of energy regularization term
    w_constr = 2.0 # w_constr >0 impose primary constraint as regularization term
    s_total = 1.0  # s_total = 1 : total Hamiltonian constraint s_total=0: standard Hamiltonian constraint
    pret = 0 # >0: start from pretrained, 0 : start from random
    
    Loss_history = []   
    Loss_eq_history= []
    Loss_en_history= []
    Loss_constr_history=[]
    
    # network training
    for ii in range(pret,6):
        plt.close('all')
        if ii == 0:
            ld = False
        else: ld = True
        t_max = (1+ii)*Tmax
        N = int((1+0.1*ii)*N)
        dt = t_max/N 
        n_train = N 
        model, loss, loss_eq, loss_erg, loss_constr,runTime = \
            train_HDNN(P0, t_max, inpn,outn,hidn, neurons, epochs, n_train, lr, loadWeights=ld, minLoss=1e-8)
    
    # save training metrics
    np.savetxt('data/loss.txt',loss)
    np.savetxt('data/loss_eq.txt',loss_eq)
    np.savetxt('data/loss_erg.txt',loss_erg)
    np.savetxt('data/loss_constr.txt',loss_constr)

    # print training time and loss
    print('Training time (minutes):', runTime/60)
    print('Training Loss: ',  loss[-1] )
    
    # generate predicted solutions
    epsilon0 =0.2
    load_data(epsilon0,w_constr)
    
