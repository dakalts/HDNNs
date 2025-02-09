#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dimitrios Kaltsas

Code for the paper:
"Constraint Hamiltonian systems and physics-informed neural networks: Hamilton-Dirac neural networks"
by D. A. Kaltsas, Physical Review E 111, 025301 (2025).

Example 1: HDNN for the Planar Pendulum

First version: 25/12/2023
Last updated: 27/01/2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import copy
import time
import sys
from os import path
import matplotlib.pyplot as plt
from pend_helpers import LSODA
from IPython import get_ipython

# Set up plotting and clear cache
get_ipython().run_line_magic('matplotlib', 'qt')
plt.close('all')
torch.cuda.empty_cache()  # Release unoccupied cached memory
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print(f"Using device: {device}")

# Custom activation function: Sinegmoid
class Sinegmoid(nn.Module):
    def __init__(self, delta1, delta2):
        super().__init__()
        self.delta1 = delta1
        self.delta2 = delta2

    def forward(self, x):
        return torch.sin(self.delta1 * x) * torch.sigmoid(self.delta2 * x)


# Auto-differentiation for derivatives
def dfx(x, f):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


# Perturb evaluation points stochastically
def perturbPoints(grid, t0, tf, sig=0.5):
    delta_t = grid[1] - grid[0]
    noise = delta_t * torch.randn_like(grid) * sig
    t = grid + noise
    t.data[2] = -1  # Force specific point
    t.data[t < t0] = t0 - t.data[t < t0]  # Ensure points are within [t0, tf]
    t.data[t > tf] = 2 * tf - t.data[t > tf]
    t.requires_grad = False
    return t


# Parametric solutions for the system
def parametricSolutions(t, nn, m, P0, gamma):
    t0, px0, py0, theta = P0[0], P0[3], P0[4], P0[5]
    x0, y0 = np.cos(theta), np.sin(theta)
    N1, N2, N3, N4 = nn(t, m)
    dt = t - t0
    f = 1 - torch.exp(-gamma * dt)  # Decay function for parametric solution
    x_hat = x0 + f * N1
    y_hat = y0 + f * N2
    px_hat = px0 + f * N3
    py_hat = py0 + f * N4
    return x_hat, y_hat, px_hat, py_hat


# Loss function for Hamilton-Dirac equations
def HamDirEqs_Loss(t, x, y, px, py, m, g):
    mu = -(x * px + y * py) / (m * (x**2 + y**2))
    lam = -(px**2 / m + py**2 / m - m * g * y) / (x**2 + y**2)
    xd, yd, pxd, pyd = dfx(t, x), dfx(t, y), dfx(t, px), dfx(t, py)
    fx = xd - px / m - mu * x
    fy = yd - py / m - mu * y
    fpx = pxd - lam * x + mu * px
    fpy = pyd - lam * y + mu * py + m * g
    L = (fx.pow(2) + fy.pow(2) + fpx.pow(2) + fpy.pow(2)).mean()
    return L


# Hamiltonian function
def hamiltonian(x0, y0, x, y, px, py, m, g):
    V = m * g * (1 + y)
    K = (px**2 + py**2) / (2 * m)
    lam = -(px**2 / m + py**2 / m - m * g * y) / (x**2 + y**2)
    ham = K + V - 0.5 * lam * (x**2 + y**2 - (x0**2 + y0**2))
    return ham


# Primary and secondary constraint losses
def prim_constr_loss(x, y, x0, y0):
    return (x**2 + y**2 - x0**2 - y0**2).pow(2).mean()


def second_constr_loss(x, y, px, py):
    return (x * px + y * py).pow(2).mean()


# HDNN architecture
class HDNN(nn.Module):
    def __init__(self, inpn, outn, hidn, D_hid, actF):
        super(HDNN, self).__init__()
        self.actF = actF
        self.Lin_1 = nn.Linear(inpn, D_hid)
        self.Lin_hid = nn.Linear(D_hid, D_hid)
        self.Lin_out = nn.Linear(D_hid, outn)

    def forward(self, t, theta):
        ones = torch.ones_like(t)
        inputs = torch.cat([t, ones * theta], axis=1)
        h = self.actF(self.Lin_1(inputs))
        for _ in range(hidn - 1):
            h = self.actF(self.Lin_hid(h))
        r = self.Lin_out(h)
        return [r[:, i].reshape(-1, 1) for i in range(outn)]


# Weight initialization
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        y = m.in_features
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        m.bias.data.fill_(0)

def train_HDNN(P0, tf, inpn, outn, hidn, neurons, epochs, n_train, lr, loadWeights, minLoss, PATH="models/model_HDNN_pend"):
    """
    Train the HDNN model for the planar pendulum system.

    Args:
        P0 (list): Initial conditions and parameters [t0, m1, m2, px0, py0, theta0, g].
        tf (float): Final time.
        inpn (int): Number of input neurons.
        outn (int): Number of output neurons.
        hidn (int): Number of hidden layers.
        neurons (int): Number of neurons per hidden layer.
        epochs (int): Number of training epochs.
        n_train (int): Number of training points.
        lr (float): Learning rate.
        loadWeights (bool): Whether to load pre-trained weights.
        minLoss (float): Minimum loss threshold for early stopping.
        PATH (str): Path to save/load the model.

    Returns:
        fc1 (HDNN): Trained model with the lowest loss.
        Loss_history (list): Total loss history.
        Loss_eq_history (list): Hamiltonian equations loss history.
        Loss_en_history (list): Energy regularization loss history.
        Loss_constr_history (list): Constraint loss history.
        L2err_history (list): L2 error history.
        runTime (float): Total training time.
    """
    # Initialize the HDNN model and optimizer
    fc0 = HDNN(inpn, outn, hidn, neurons, actF).to(device)
    fc0.apply(weights_init_normal)  # Initialize weights
    fc1 = copy.deepcopy(fc0)  # Deep copy for the best model
    optimizer = optim.Adam(list(fc0.parameters()) + [gamma, delta1,delta2], lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.1, total_steps=epochs)

    # Load pre-trained weights if specified
    if loadWeights and path.exists(PATH):
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        fc0.train()

    # Extract initial conditions
    t0, m1, m2, px0, py0, theta0, g = P0
    x0, y0 = np.cos(theta0), np.sin(theta0)
    grid = torch.linspace(t0, tf, n_train).reshape(-1, 1).to(device)
    m0 = (m1 + m2) / 2

    # Numerical solution for comparison
    t_num = np.linspace(t0, tf, n_train)
    x_num, y_num, px_num, py_num = LSODA(n_train,t_num, x0, y0, px0, py0, m0, g)

    # Training loop
    Loss_history, Loss_eq_history, Loss_en_history, Loss_constr_history, L2err_history = [], [], [], [], []
    Llim = 1  # Threshold for saving the best model
    TeP0 = time.time()

    plt.figure(figsize=(10, 10))
    for tt in range(epochs):
        # Perturb evaluation points
        t = perturbPoints(grid, t0, tf, sig=0.3 * tf).to(device)
        t.requires_grad = True

        # Sample mass parameter
        if tt % 10 == 0:
            m = torch.from_numpy(np.random.beta(0.99, 0.99, size=(n_train, 1))).float()
            m = (m1 + m * (m2 - m1)).to(device)
            ham0 = hamiltonian(x0, y0, x0, y0, px0, py0, m, g)

        # Network solutions
        x, y, px, py = parametricSolutions(t, fc0, m, P0, gamma)

        # Loss components
        L_eq = HamDirEqs_Loss(t, x, y, px, py, m, g)
        ham = hamiltonian(x0, y0, x, y, px, py, m, g)
        L_en = 0.5 * ((ham - ham0).pow(2)).mean()
        L_constr = prim_constr_loss(x, y, x0, y0) + second_constr_loss(x, y, px, py)

        # L2 error calculation
        x_, y_, px_, py_ = parametricSolutions(grid, fc0, m0, P0, gamma)
        dx_ = x_[:n_train, 0] - torch.from_numpy(x_num).to(device)
        dy_ = y_[:n_train, 0] - torch.from_numpy(y_num).to(device)
        dpx_ = px_[:n_train, 0] - torch.from_numpy(px_num).to(device)
        dpy_ = py_[:n_train, 0] - torch.from_numpy(py_num).to(device)
        L2err = torch.sqrt((dx_**2 + dy_**2 + dpx_**2 + dpy_**2).sum()) / n_train

        # Total loss
        Ltot = w_eq * L_eq + w_en * L_en + w_constr * L_constr
        Ltot.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log losses and errors
        Loss_history.append(Ltot.item())
        Loss_eq_history.append(L_eq.item())
        Loss_en_history.append(L_en.item())
        Loss_constr_history.append(L_constr.item())
        L2err_history.append(L2err.item())

        # Save the best model
        if tt > 0.8 * epochs and Ltot < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = Ltot

        # Early stopping
        if Ltot < minLoss:
            print('Reached minimum requested loss')
            break

        # Print progress and visualize
        if tt % 50 == 0:
            print(f"Epoch: {tt}, L_eq = {L_eq:.5f}, L_en = {L_en:.5f}, L_constr = {L_constr:.5f}, \
                  L2err = {L2err:.5f}, lr = {lr:.5f},\
                  gamma = {gamma:.5f}, delta1 = {delta1:.5f}, delta2 = {delta2:.5f}\
                      --------------------------------------------------------------")
            plt.clf()
            plt.scatter(x_[:40].detach().cpu().numpy(), y_[:40].detach().cpu().numpy(), s=3, label='HDNN')
            plt.scatter(x_num[:40], y_num[:40], s=3, label='Exact')
            plt.xlabel('x', fontsize=14)
            plt.ylabel('y', fontsize=14, rotation=0)
            plt.gca().set_aspect('equal')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)
            plt.pause(0.1)
            plt.show()

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

    # Save the final model
    torch.save({
        'epoch': tt,
        'model_state_dict': fc1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': Ltot,
    }, PATH)

    runTime = time.time() - TeP0
    return fc1, Loss_history, Loss_eq_history, Loss_en_history, Loss_constr_history, L2err_history, runTime

def loadModel(PATH="models/model_HDNN_pend"):
    if path.exists(PATH):
        fc0 = HDNN(inpn,outn,hidn,neurons,actF)
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train() # or model.eval
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()

    return fc0.to(device)

def loss_plots(loss, loss_eq, loss_erg, loss_constr, l2err, w_constr):
    """
    Plots various loss metrics with logarithmic scaling.
    
    variables:
    loss (array-like): Total loss values over epochs.
    loss_eq (array-like): Equation loss values.
    loss_erg (array-like): Energy loss values.
    loss_constr (array-like): Constraint loss values.
    l2err (array-like): L2 norm error values.
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
    mean_l2err, std_l2err = group_and_calculate_stats(l2err)
    
    epochs = np.arange(len(mean_loss))  # X-axis values
    
    plt.figure(figsize=(10, 8))
    
    # Plot mean losses with different styles
    plt.loglog(mean_loss, 'b', alpha=0.6, linewidth=2, label='Total loss')
    plt.loglog(mean_loss_eq, 'c--', alpha=0.6, linewidth=2, label='Equation loss')
    plt.loglog(mean_loss_erg, 'r:', alpha=0.6, linewidth=3, label='Energy loss')
    plt.loglog(mean_loss_constr, 'g-.', alpha=0.6, linewidth=2, label='Constraint loss')
    plt.loglog(mean_l2err, 'y', marker='o', ms=3, mfc='y', linewidth=2, alpha=0.6, label='Norm L2 error')
    
    # Fill between mean ± std for visualization of variance
    plt.fill_between(epochs, mean_loss + std_loss, mean_loss - std_loss, facecolor='blue', alpha=0.35)
    plt.fill_between(epochs, mean_loss_eq + std_loss_eq, mean_loss_eq - std_loss_eq, facecolor='cyan', alpha=0.35)
    plt.fill_between(epochs, mean_loss_erg + std_loss_erg, mean_loss_erg - std_loss_erg, facecolor='red', alpha=0.35)
    plt.fill_between(epochs, mean_loss_constr + std_loss_constr, mean_loss_constr - std_loss_constr, facecolor='green', alpha=0.35)
    plt.fill_between(epochs, mean_l2err + std_l2err, mean_l2err - std_l2err, facecolor='orange', alpha=0.35)
    
    # Configure plot labels and legend
    plt.legend(fontsize=20)
    plt.ylabel('Loss', fontsize=22)
    plt.xlabel(r'Epochs $(x10^2)$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Save the figure
    plt.savefig(f'figs/losses_pend_hd_{w_constr}.pdf', format="pdf", bbox_inches="tight")
    plt.show()


def load_data(theta0, w_constr):
    """
    Loads model and loss data from files, then generates loss plots.
    
    Parameters:
    theta0 (float): Initial angle.
    w_constr (float): Constraint weight used in filenames.
    """
    model = loadModel()
    
    # Load loss data from text files
    loss = np.loadtxt(f'data/loss_{w_constr}.txt')
    loss_eq = np.loadtxt(f'data/loss_eq_{w_constr}.txt')
    loss_erg = np.loadtxt(f'data/loss_erg_{w_constr}.txt')
    loss_constr = np.loadtxt(f'data/loss_constr_{w_constr}.txt')
    l2err = np.loadtxt(f'data/l2err_{w_constr}.txt')
    
    from pred_solutions import pred_solutions
    # Generate prediction
    pred_solutions(theta0, P0, N, t_max, model, device, gamma, n_train, Tpend, w_constr, dt)
    
    # Plot loss data
    loss_plots(loss, loss_eq, loss_erg, loss_constr, l2err, w_constr)
    
    
if __name__ == "__main__":

    # training range for mass parameter
    m1, m2 =1.0 , 2.0
    
    # Initial conditions and fixed parameters
    t0, theta0, px0, py0,  g =  0, -0.15, 0.0, 0.0, 10; 
    P0 = [t0,m1,m2, px0, py0, theta0, g]
    
    
    # pendulum period
    import scipy.integrate as integrate
    
    kk = np.sin((np.pi/2)/2 ) 
    Tpend = 4/(np.sqrt(g))*integrate.quad(lambda x: 1/((np.sqrt(1-x**2)*(1-kk**2*x**2))), 0, 1)[0]

    # model and training parameters
    t_max = 30*Tpend
    N =  1000
    dt = t_max/N 
    n_train = N 
    inpn = 2
    outn=4
    hidn=4
    neurons = 160
    epochs = 20000
    lr = 5e-3
    
    # trainable parameters
    gamma = nn.Parameter(torch.tensor(7.0),requires_grad=True)
    delta1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
    delta2 = nn.Parameter(torch.tensor(1.0),requires_grad=True)
    
    #activation funcction
    actF= Sinegmoid(delta1,delta2)  # Sinegmoid(), nn.Sigmoid(), nn.SiLU, nn.Tanh
    
    w_eq=1.0     # weight of equation residual term
    w_en=1.0     # weight of energy regularization term
    w_constr=10.0 # w_constr >0 impose primary constraint as regularization term
    s_total=1.0  # s_total = 1 : total Hamiltonian constraint s_total=0: standard Hamiltonian constraint
    ld = False   # load weights of pretrained network

    # network training
    model, loss, loss_eq, loss_erg, loss_constr, l2err, runTime = \
        train_HDNN(P0, t_max, inpn,outn,hidn, neurons, epochs, n_train, lr, loadWeights=ld, minLoss=1e-8)
    
    # save training metrics
    np.savetxt('data/loss_'+ str(w_constr)+'.txt',loss)
    np.savetxt('data/loss_eq_'+ str(w_constr)+'.txt',loss_eq)
    np.savetxt('data/loss_erg_'+ str(w_constr)+'.txt',loss_erg)
    np.savetxt('data/loss_constr_'+ str(w_constr)+'.txt',loss_constr)
    np.savetxt('data/l2err_'+ str(w_constr)+'.txt',l2err)

    # print training time and loss
    print('Training time (minutes):', runTime/60)
    print('Training Loss: ',  loss[-1] )
    
    # generate predicted solutions
    m0 = 1.5
    load_data(m0,w_constr)


