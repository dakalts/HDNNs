"""

@author: Dimitrios Kaltsas

Code for the paper:
"Constraint Hamiltonian systems and physics-informed neural networks: Hamilton-Dirac neural networks"
by D. A. Kaltsas, Physical Review E 111, 025301 (2025).

Example 3: HDNN for the Guiding center motion
in magnetic field
B = B(x,y) \hat{z}
B(x,y) = 1+ epsilon(x^2/kappa + y^2)

First version: 25/12/2023
Last updated: 27/01/2025
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from os import path
import sys
from gc_helpers import RK45, LSODA, energy
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

    def forward(self, input):
        return torch.sin(self.delta1 * input) * torch.sigmoid(self.delta2 * input)

# Compute derivative using auto-differentiation
def dfx(x, f):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

# Perturb evaluation points to introduce stochasticity
def perturbPoints(grid, t0, tf, sig=0.5):
    delta_t = grid[1] - grid[0]  # Compute time step
    noise = delta_t * torch.randn_like(grid) * sig
    t = grid + noise
    
    # Enforce boundary conditions
    t.data[2] = -1
    t.data[t < t0] = 2 * t0 - t.data[t < t0]
    t.data[t > tf] = 2 * tf - t.data[t > tf]
    return t

# Define parametric solutions
def parametricSolutions(t, kappa, nn, P0, gamma):
    t0, x0, y0, px0, py0 = P0[:5]
    N1, N2, N3, N4 = nn(t, kappa)
    dt = t - t0
    f = 1 - torch.exp(-gamma * dt)  # Activation function
    
    # Compute solutions
    x_hat = x0 + f * N1
    y_hat = y0 + f * N2
    px_hat = px0 + f * N3
    py_hat = py0 + f * N4
    return x_hat, y_hat, px_hat, py_hat

# Define Hamilton-Dirac loss function
def HamDirEqs_Loss(t, x, y, px, py, mu, epsilon, kappa):
    B = 1 + epsilon * (x**2 / kappa + y**2)
    Wx = 2 * mu * epsilon * x / kappa
    Wy = mu * epsilon * 2 * y
    xd, yd, pxd, pyd = dfx(t, x), dfx(t, y), dfx(t, px), dfx(t, py)
    
    Ax = -0.5 * (y + epsilon * (x**2 * y / kappa + y**3 / 3))
    Ay = 0.5 * (x + epsilon * (x**3 / (3 * kappa) + y**2 * x))
    
    fx = xd - ((px - Ax) - 2 * epsilon * mu * y / B)
    fy = yd - ((py - Ay) + 2 * epsilon * mu * x / (kappa * B))
    fpx = pxd - (-epsilon * x * y / kappa * xd - 0.5 * (1 + epsilon * (x**2 / kappa + y**2)) * yd)
    fpy = pyd - (0.5 * (1 + epsilon * (x**2 / kappa + y**2)) * xd + epsilon * y * x * yd)
    
    loss = (fx.pow(2) + fy.pow(2) + fpx.pow(2) + fpy.pow(2)).mean()
    constraint_loss = (Wx * xd + Wy * yd).pow(2).mean()
    return loss + constraint_loss

# Define the Hamiltonian function
def hamiltonian(x, y, px, py, mu, epsilon, kappa):
    Ax = -0.5 * (y + epsilon * (x**2 * y / kappa + y**3 / 3))
    Ay = 0.5 * (x + epsilon * (x**3 / (3 * kappa) + y**2 * x))
    B = 1 + epsilon * (x**2 / kappa + y**2)
    W = mu * B
    Wx = 2 * mu * epsilon * x / kappa
    Wy = mu * epsilon * 2 * y
    return W - (px - Ax) * Wy / B + (py - Ay) * Wx / B

# Define primary constraint loss
def prim_constr_loss(t, x, y, px, py, epsilon, kappa, mu):
    Ax = -0.5 * (y + epsilon * (x**2 * y / kappa + y**3 / 3))
    Ay = 0.5 * (x + epsilon * (x**3 / (3 * kappa) + y**2 * x))
    phi1 = px - Ax
    phi2 = py - Ay
    return phi1.pow(2).mean() + phi2.pow(2).mean()

# Define neural network architecture
class HDNN(nn.Module):
    def __init__(self, inpn, outn, hidn, D_hid, actF):
        super(HDNN, self).__init__()
        self.actF = actF
        self.Lin_1 = nn.Linear(inpn, D_hid)
        self.Lin_hid = nn.Linear(D_hid, D_hid)
        self.Lin_out = nn.Linear(D_hid, outn)

    def forward(self, t, kappa):
        ones = torch.ones_like(t)
        inputs = torch.cat([t, ones * kappa], axis=1)
        h = self.actF(self.Lin_1(inputs))
        h = self.actF(self.Lin_hid(h))
        r = self.Lin_out(h)
        return [r[:, i].reshape(-1, 1) for i in range(r.shape[1])]

# Initialize network weights
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        y = m.in_features
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        m.bias.data.fill_(0)
        
# Train the Neural Network
def train_HDNN(P0, tf, inpn, outn, hidn, neurons, epochs, n_train, lr, loadWeights, minLoss, PATH="models/model_HDNN_gc"):
    
    # Initialize the HDNN model
    fc0 = HDNN(inpn, outn, hidn, neurons, actF).to(device)
    fc0.apply(weights_init_normal)  # Initialize weights using a normal distribution
    
    # Keep a copy of the best model (lowest training loss)
    fc1 = copy.deepcopy(fc0)
    
    # Extract parameters from P0
    t0, x0, y0, px0, py0, mu0, epsilon, kappa1, kappa2 = P0

    # Define optimizer with model parameters and additional trainable parameters
    optimizer = optim.Adam(list(fc0.parameters()) + [gamma, delta1, delta2, mu], lr=lr)
    Llim = 1  # Initial loss threshold for keeping the best model
    
    # Create a grid of training points
    grid = torch.linspace(t0, tf, n_train).reshape(-1, 1).to(device)
    
    # Load pre-trained weights if specified
    if loadWeights:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tt = checkpoint['epoch']
        Ltot = checkpoint['loss']
        fc0.train()  # Set model to training mode
    
    # Compute mean kappa value
    kappam = torch.tensor((kappa1 + kappa2) / 2)
    
    # Generate numerical solutions using LSODA solver
    t_num = np.linspace(t0, tf, 500)
    x_num, y_num, px_num, py_num = LSODA(500, t_num, x0, y0, px0, py0, mu0, epsilon, kappam)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.1, total_steps=epochs)

    # Start training
    TeP0 = time.time()  # Start time for training
    for tt in range(epochs):
        
        # Perturb the evaluation points and ensure t[0] = t0
        t = perturbPoints(grid, t0, tf, sig=0.03 * tf).to(device)
        t.requires_grad = True  # Enable gradient computation

        # Sample random kappa values from a Beta distribution
        if tt % 10 == 0:
            kappa = torch.from_numpy(np.random.beta(0.99, 0.99, size=(n_train, 1))).float()
            kappa = (kappa1 + kappa * (kappa2 - kappa1)).to(device)
            ham0 = hamiltonian(x0, y0, px0, py0, mu, epsilon, kappa)  # Compute initial Hamiltonian

        # Compute parametric solutions for validation
        x_, y_, px_, py_ = parametricSolutions(torch.linspace(t0, tf, 500).reshape(-1, 1).to(device), kappam, fc0, P0, gamma)

        # Compute L2 error between numerical and learned solutions
        dx_ = (x_[:, 0] - torch.from_numpy(x_num).to(device))
        dy_ = (y_[:, 0] - torch.from_numpy(y_num).to(device))
        dpx_ = (px_[:, 0] - torch.from_numpy(px_num).to(device))
        dpy_ = (py_[:, 0] - torch.from_numpy(py_num).to(device))
        L2err = torch.sqrt((dx_**2 + dy_**2 + dpx_**2 + dpy_**2).sum()) / 500
        l2err = L2err.detach().cpu().numpy()
        Ld = (dx_**2 + dy_**2 + dpx_**2 + dpy_**2).mean()

        # Compute network solutions
        x, y, px, py = parametricSolutions(t, kappa, fc0, P0, gamma)

        # Compute loss components
        L_eq = HamDirEqs_Loss(t, x, y, px, py, mu, epsilon, kappa)  # Hamiltonian equation loss
        
        # Energy regularization loss
        ham = hamiltonian(x, y, px, py, mu, epsilon, kappa)
        L_en = 0.5 * ((ham - ham0).pow(2)).mean()

        # Constraint loss (for stabilization)
        if tt < 1:
            L_constr = L_en
        else:
            L_constr = prim_constr_loss(t, x, y, px, py, epsilon, kappa, mu)

        # Total loss function
        Ltot = (w_eq * L_eq + w_en * L_en + w_constr * L_constr) + w_dat * Ld

        # Optimize model
        Ltot.backward(retain_graph=False)  # Compute gradients
        optimizer.step()  # Update model parameters
        optimizer.zero_grad()  # Reset gradients
        
        # Track loss history
        Loss_history.append(Ltot.data.cpu().numpy())
        Loss_eq_history.append(L_eq.data.cpu().numpy())
        Loss_en_history.append(L_en.data.cpu().numpy())
        Loss_constr_history.append(L_constr.data.cpu().numpy())
        L2err_history.append(l2err)
        dmu.append(abs((mu - mu0) / mu0).data.cpu().numpy())

        # Keep the best model (lowest L2 error)
        if tt > 0.5 * epochs and L2err < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = L2err

        # Stop training if loss threshold is met
        if Ltot < minLoss:
            fc1 = copy.deepcopy(fc0)
            print('Reached minimum requested loss.')
            break

        # Print progress every 50 epochs
        if (tt + 1) % 50 == 0:
            print(f"Epoch: {tt + 1}, L_eq = {L_eq:.5f}, L_en = {L_en:.5f}, L_constr = {L_constr:.5f}, \n\
              lr = {lr:.3f}, gamma = {gamma:.3f}, delta1 = {delta1:.3f}, delta2 = {delta2:.3f}, mu = {mu:.3f},\n\
              epsilon = {epsilon:.3f}, L2err = {L2err:.5f}, N = {n_train}, tmax = {t_max / np.pi}\n\
                  -----------------------------------------------------------")
            
            # Plot training progress
            plt.clf()
            nsp = 120
            plt.scatter(x_[0:nsp].detach().cpu().numpy(), y_[0:nsp].detach().cpu().numpy(), s=3, label='HDNN')
            plt.scatter(x_num[0:nsp], y_num[0:nsp], s=3, label='LSODA')
            plt.ylabel('y', rotation=0, fontsize=14)
            plt.xlabel('x', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.gca().set_aspect('equal')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)
            plt.pause(0.1)
            plt.show()

        # Adjust learning rate
        if sched:
            scheduler.step()
            lr = scheduler.get_lr()[0]

    # Calculate total runtime
    TePf = time.time()
    runTime = TePf - TeP0

    # Save the trained model
    torch.save({
        'epoch': tt,
        'model_state_dict': fc1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': Ltot,
    }, PATH)

    return fc1, Loss_history, Loss_eq_history, Loss_en_history, Loss_constr_history, L2err_history, runTime, dmu


# Function to load a trained model
def loadModel(PATH="models/model_HDNN_gc"):
    if path.exists(PATH):
        fc0 = HDNN(inpn, outn, hidn, neurons, actF)
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train()  # Set to training mode
    else:
        print('Warning: No trained model found. Terminating.')
        sys.exit()

    return fc0.to(device)



def loss_plots(loss,loss_eq,loss_erg,loss_constr, l2err, dmu):
    """
    Plots various loss metrics with logarithmic scaling.
    
    variables:
    loss (array-like): Total loss values over epochs.
    loss_eq (array-like): Equation loss values.
    loss_erg (array-like): Energy loss values.
    loss_constr (array-like): Constraint loss values.
    l2err (array-like): L2 norm error values.
    dmu (array-like): \delta \mu values.
    """
    groupsize = 100  # Number of epochs per averaging group
    # Group losses into chunks of 'groupsize' for averaging
    
    def group_and_calculate_stats(data):
        grouped = np.array([data[x:x+groupsize] for x in range(0, len(data), groupsize)])
        return np.array([group.mean() for group in grouped]), np.array([group.std() for group in grouped])
    
    mean_loss, std_loss = group_and_calculate_stats(loss)
    mean_loss_eq, std_loss_eq = group_and_calculate_stats(loss_eq)
    mean_loss_erg, std_loss_erg = group_and_calculate_stats(loss_erg)
    mean_loss_constr, std_loss_constr = group_and_calculate_stats(loss_constr)
    mean_l2err, std_l2err = group_and_calculate_stats(l2err)
    mean_dmu, std_dmu = group_and_calculate_stats(l2err)
    
    xm = np.arange(len(mean_loss))  # X-axis values
    plt.figure()
    plt.loglog(mean_loss,'b',alpha=0.6, linewidth=2, label='Total loss')
    plt.loglog(mean_loss_eq,'c--',alpha=0.6,linewidth=2, label='Equation loss')
    plt.loglog(mean_loss_erg,'r:',alpha=0.6,linewidth=3, label='Energy loss')
    plt.loglog(mean_loss_constr, 'g-.', alpha=0.6,linewidth=2, label='Constraint loss')
    plt.loglog(mean_l2err,'y',marker='o', ms = 4, mfc = 'y', alpha=0.6, label='norm L2 error')
    if w_dat!=0.0:
        plt.loglog(mean_dmu,'m',marker='x', ms = 4, mfc = 'm', alpha=0.6, label=r'$\epsilon_\mu$')
        plt.fill_between(xm,mean_dmu+std_dmu, mean_dmu-std_dmu, facecolor='magenta', alpha=0.35)
    plt.fill_between(xm,mean_loss+std_loss, mean_loss-std_loss, facecolor='blue', alpha=0.35)
    plt.fill_between(xm,mean_loss_eq+std_loss_eq, mean_loss_eq-std_loss_eq, facecolor='cyan', alpha=0.35)
    plt.fill_between(xm,mean_loss_erg+std_loss_erg, mean_loss_erg-std_loss_erg, facecolor='red', alpha=0.35)
    plt.fill_between(xm,mean_loss_constr+std_loss_constr, mean_loss_constr-std_loss_constr, facecolor='green', alpha=0.35)
    plt.fill_between(xm,mean_l2err+std_l2err, mean_l2err-std_l2err, facecolor='orange', alpha=0.35)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.tight_layout()
    plt.ylabel('Loss',fontsize=20,rotation=90)
    plt.xlabel(r'epochs $(x10^2)$',fontsize=20,rotation=0)    
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.savefig('figs/losses_hd_' + str(w_constr) + '.pdf',format="pdf", bbox_inches="tight")
     

def load_data(kappa0, w_constr):
    """ Load trained model and loss history, then generate predictions. """
    
    model = loadModel()

    # Load loss history from saved files
    loss = np.loadtxt(f'data/loss_{w_constr}.txt')
    loss_eq = np.loadtxt(f'data/loss_eq_{w_constr}.txt')
    loss_erg = np.loadtxt(f'data/loss_erg_{w_constr}.txt')
    loss_constr = np.loadtxt(f'data/loss_constr_{w_constr}.txt')
    l2err = np.loadtxt(f'data/l2err_{w_constr}.txt')
    dmu = np.loadtxt(f'data/dmu_{w_constr}.txt')

    from pred_solutions import pred_solutions
    # Generate predictions
    pred_solutions(kappa0, P0, model, n_train, t_max, gamma, mu.item(), w_constr, dt)

    # Plot the losses
    loss_plots(loss, loss_eq, loss_erg, loss_constr, l2err, dmu)


if __name__ == "__main__":
    
    """ Main script for training and evaluating the model. """
   
    # training range for kappa parameter
    kappa1, kappa2 = 3.8, 4.2
    kappa0 = (kappa1 + kappa2) / 2
    
    # gc and magnetic field parameters 
    mu0 = 2.0
    epsilon = 0.15
    
    # Initial conditions and fixed parameters
    theta0 = np.pi / 2
    t0, x0, y0 = 0, np.cos(theta0), np.sin(theta0)
    Ax0 = -0.5 * (y0 + epsilon * (x0**2 * y0 / kappa0 + y0**3 / 3))
    Ay0 = 0.5 * (x0 + epsilon * (x0**3 / (3 * kappa0) + y0**2 * x0))
    px0, py0 = Ax0, Ay0
    P0 = [t0, x0, y0, px0, py0, mu0, epsilon, kappa1, kappa2]
    
    # Simulation parameters
    tmax = 50 * np.pi
    N = 500
    inpn = 2
    outn = 4
    hidn = 4
    neurons = 160
    epochs = 1000
    lr = 20e-3
    sched = False
    
    # Model hyperparameters
    gamma = nn.Parameter(torch.tensor(7.0), requires_grad=True)
    delta1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    delta2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    actF = Sinegmoid(delta1, delta2)

    # Loss function weights
    w_eq = 1.0
    w_en = 1.0
    w_constr = 1.0
    w_dat = 0.0     # data-driven parameter inference 
    s_total = 1.0  # 1: total Hamiltonian constraint, 0: standard Hamiltonian constraint

    # Trainable parameter
    if w_dat==0:
       mu = nn.Parameter(torch.tensor(mu0),requires_grad=False)

    else  : 
       mu = nn.Parameter(torch.tensor(0.0),requires_grad=True)


    # Training history lists
    Loss_history, Loss_eq_history, Loss_en_history = [], [], []
    Loss_constr_history, L2err_history, dmu = [], [], []

    """ Train the neural network """
    
    pret = 0  # 0: train from scratch, >0: start from a pretrained model
    Nex = 7  # Number of training iterations

    for ii in range(pret, Nex):
        plt.close('all')
        if ii == 0:
            ld = False
        else: 
            ld = True
        
        # Adjust training parameters for each iteration
        t_max = (1 + 0.5 * ii) * tmax
        n_train = int((1 + 0.5 * ii) * N)
        epochs = int((1 + 0.1) * epochs)
        dt = t_max / n_train

        # Train model
        model, loss, loss_eq, loss_erg, loss_constr, l2err, runTime, dmu = \
            train_HDNN(P0, t_max, inpn, outn, hidn, neurons, epochs, n_train, lr, loadWeights=ld, minLoss=1e-8)
   
    # save training metrics
    np.savetxt('data/loss_'+ str(w_constr)+'.txt',loss)
    np.savetxt('data/loss_eq_'+ str(w_constr)+'.txt',loss_eq)
    np.savetxt('data/loss_erg_'+ str(w_constr)+'.txt',loss_erg)
    np.savetxt('data/loss_constr_'+ str(w_constr)+'.txt',loss_constr)
    np.savetxt('data/l2err_'+ str(w_constr)+'.txt',l2err)
    np.savetxt('data/dmu_'+ str(w_constr)+'.txt',dmu)

    # print training time and loss
    print('Training time (minutes):', runTime / 60)
    print('Final Training Loss:', loss[-1])

    """ Load trained model and generate predictions """
    kappa0 = 4.0
    load_data(kappa0, w_constr)