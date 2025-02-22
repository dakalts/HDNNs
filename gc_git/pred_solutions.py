import numpy as np
import torch
from HDNN_GC_multi_kappa import *
from gc_helpers import RK45, LSODA, energy, saveData
import matplotlib.pyplot as plt
from IPython import get_ipython
import time

# Enable interactive plotting mode
get_ipython().run_line_magic('matplotlib', 'qt')

def pred_solutions(kappa0, P0, model, n_train, t_max, gamma, mu, w_constr, dt):
    """
    Predicts solutions using a neural network model and compares them with numerical solvers.
    
    Parameters:
    kappa0 : float - Parameter for the system
    P0 : list - Initial conditions [t0, x0, y0, px0, py0, mu0, epsilon]
    model : PyTorch model - Neural network model for predictions
    n_train : int - Number of training points
    t_max : float - Maximum time for integration
    gamma : float - Parameter for system dynamics
    mu : float - Parameter for system dynamics
    w_constr : float - Weight constraint
    dt : float - Time step size
    """
    plt.close('all')
    
    # Extract initial conditions
    t0, x0, y0, px0, py0, mu0, epsilon = P0[0], P0[1], P0[2], P0[3], P0[4], P0[5], P0[6]

    # Define test time points
    nTest = 10*n_train 
    tTest = torch.linspace(t0,t_max,nTest).to(device) 
    tTest = tTest.reshape(-1,1)
    t_net = tTest.detach().cpu().numpy()

    
    x, y, px, py = parametricSolutions(tTest,torch.tensor([[kappa0]]).float().to(device),model,P0,gamma)
    x = x.data.cpu().numpy()
    y=y.data.cpu().numpy()
    px = px.data.cpu().numpy()
    py=py.data.cpu().numpy()
    
    # Compute vector potential components
    Ax = -0.5 * (y + epsilon * (x**2 * y / kappa0 + y**3 / 3))
    Ay = 0.5 * (x + epsilon * (x**3 / (3 * kappa0) + y**2 * x))
    
    # Compute energy
    E = energy(x, y, px, py, mu, epsilon, kappa0)
    
    # Solve using LSODA numerical method
    t_num = np.linspace(t0, t_max, nTest)
    x_num, y_num, px_num, py_num = LSODA(nTest, t_num, x0, y0, px0, py0, mu, epsilon, kappa0)
    E_num = energy(x_num, y_num, px_num, py_num, mu, epsilon, kappa0)
    
    Ax_num = -0.5 * (y_num + epsilon * (x_num**2 * y_num / kappa0 + y_num**3 / 3))
    Ay_num = 0.5 * (x_num + epsilon * (x_num**3 / (3 * kappa0) + y_num**2 * x_num))
    
    # Solve using RK45 numerical method
    startT_RK45 = time.time()
    Ns = nTest - 1  # Number of steps
    E_s, x_s, y_s, px_s, py_s, t_s = RK45(Ns, x0, y0, px0, py0, t0, t_max, mu, epsilon, kappa0)
    
    Ax_s = -0.5 * (y_s + epsilon * (x_s**2 * y_s / kappa0 + y_s**3 / 3))
    Ay_s = 0.5 * (x_s + epsilon * (x_s**3 / (3 * kappa0) + y_s**2 * x_s))
    
    endT_RK45 = time.time()
    runTimeRK45 = endT_RK45 - startT_RK45
    print(f'RK45 runtime is {runTimeRK45 / 60:.2f} minutes')
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot x vs time
    plt.subplot(2, 1, 1)
    plt.plot(t_net[int(nTest/2):], x[int(nTest/2):], 'b', alpha=0.85, linewidth=3, label='HDNN')
    plt.plot(t_s[int(nTest/2):], x_s[int(nTest/2):], ':g', alpha=0.95, linewidth=3, label='RK45')
    plt.plot(t_num[int(nTest/2):], x_num[int(nTest/2):], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.ylabel('$x$', rotation=0, fontsize=22)
    plt.xlabel('$t$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    
    # Plot y vs time
    plt.subplot(2, 1, 2)
    plt.plot(t_net[int(nTest/2):], y[int(nTest/2):], 'b', alpha=0.85, linewidth=3, label='HDNN')
    plt.plot(t_s[int(nTest/2):], y_s[int(nTest/2):], ':g', alpha=0.95, linewidth=3, label='RK45')
    plt.plot(t_num[int(nTest/2):], y_num[int(nTest/2):], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.ylabel('$y$', rotation=0, fontsize=22)
    plt.xlabel('$t$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'figs/gc_timeseries_hd_{w_constr}{kappa0}.pdf', format="pdf", bbox_inches="tight")
    
    ''' Energy Calculation '''
    # Compute the initial energy E0
    B0 = (1 + epsilon * (x0**2 / kappa0 + y0**2))
    E0 = mu * B0
    
    # Compute the relative energy differences
    dE = (E - E0) / E0
    dE_s = (E_s - E0) / E0
    
    # Grouping data into segments for statistical analysis
    groupsize = 100
    groups_dE = np.array([dE[x:x+groupsize] for x in range(0, len(dE), groupsize)])
    groups_dE_s = np.array([dE_s[x:x+groupsize] for x in range(0, len(dE_s), groupsize)])
    
    # Calculate mean and standard deviation for each group
    mean_dE = np.array([0.0] + [group.mean() for group in groups_dE[:-1]])
    std_dE = np.array([group.std() for group in groups_dE])
    mean_dE_s = np.array([group.mean() for group in groups_dE_s])
    std_dE_s = np.array([group.std() for group in groups_dE_s])
    
    # Time array for plotting
    NdE = len(mean_dE)
    tm = np.array([i for i in range(len(mean_dE))]) * t_max / nTest
    
    # Plot relative energy change
    plt.figure(figsize=(10,6))
    plt.plot(tm[:NdE], mean_dE[:NdE], 'b', linewidth=2, label='HDNN')
    plt.plot(tm, np.zeros_like(mean_dE), '--k', linewidth=2, label='Exact')
    plt.plot(tm, mean_dE_s, ':g', linewidth=3, label='RK45')
    
    # Fill areas between standard deviations
    plt.fill_between(tm[:NdE], mean_dE[:NdE] + std_dE[:NdE], mean_dE[:NdE] - std_dE[:NdE], facecolor='blue', alpha=0.35)
    plt.fill_between(tm[:NdE], mean_dE_s[:NdE] + std_dE_s[:NdE], mean_dE_s[:NdE] - std_dE_s[:NdE], facecolor='green', alpha=0.35)
    
    # Labels and legend
    plt.ylabel('$\Delta E/E_0$', rotation=0, fontsize=22)
    plt.xlabel('$t$ ($\times10^2$)', rotation=0, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/gc_energy_hd_{w_constr}{kappa0}.pdf', format='pdf', bbox_inches='tight')
    
    ''' Constraints Analysis '''
    # Compute primary and secondary constraints
    L2Phi1 = abs(px - Ax)
    L2Phi1_s = abs(px_s - Ax_s)
    L2Phi2 = abs(py - Ay)
    L2Phi2_s = abs(py_s - Ay_s)
    
    plt.figure(figsize=(10,10))
    
    # Plot first constraint
    plt.subplot(2,1,1)
    plt.plot(t_net, L2Phi1, 'b', linewidth=2, alpha=0.75, label='HDNN')
    plt.plot(t_s, L2Phi1_s, ':g', linewidth=3, alpha=0.85, label='RK45')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.ylabel(r'$|\Phi_1|$', rotation=0, fontsize=22)
    plt.xlabel(r't', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.legend(bbox_to_anchor=(1.0, 0.5), fontsize=20)
    
    # Plot second constraint
    plt.subplot(2,1,2)
    plt.plot(t_net, L2Phi2, 'b', linewidth=2, alpha=0.75, label='HDNN')
    plt.plot(t_s, L2Phi2_s, ':g', linewidth=3, alpha=0.85, label='RK45')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.ylabel(r'$|\Phi_2|$', rotation=0, fontsize=22)
    plt.xlabel(r't', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.legend(bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.savefig(f'figs/gc_L2constr_hd_{w_constr}{kappa0}.pdf', format='pdf', bbox_inches='tight')
    
    ''' Orbit Drift Analysis '''
    # Compute trajectory error
    drhdnn = abs(1 - (x**2/kappa0 + y**2))
    drrk45 = abs(1 - (x_s**2/kappa0 + y_s**2))
    plt.figure(figsize=(10,8))
    plt.plot(t_net, drhdnn, 'b', linewidth=2, label='HDNN')
    plt.plot(t_s, drrk45, ':g', linewidth=2, label='RK45 x10')
    plt.ylabel('$|\Delta r|$', rotation=90, fontsize=22)
    plt.xlabel('$t$', rotation=0, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/gc_trajectory_error_hd_{w_constr}{kappa0}.pdf', format='pdf', bbox_inches='tight')
    
    


    # Number of exact solution points
    Nexact = 800
    
    
    # GC x-y Trajectory Plot
    plt.figure(figsize=(10, 10))
    
    # First subplot: Full trajectory comparison
    plt.subplot(2, 1, 1)
    plt.plot(x[0:nTest], y[0:nTest], 'b', linewidth=4, alpha=0.75, label='HDNN')  # HDNN trajectory
    plt.plot(x_s, y_s, ':g', linewidth=1, alpha=0.85, label='RK45')  # RK45 trajectory
    plt.plot(x_num[0:Nexact], y_num[0:Nexact], '--r', alpha=0.85, linewidth=1.5, label='LSODA')  # LSODA trajectory
    plt.ylabel('$y$', rotation=0, fontsize=22)
    plt.xlabel('$x$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.gca().set_aspect('equal')
    
    # Second subplot: Zoomed-in trajectory
    plt.subplot(2, 1, 2)
    plt.plot(x[0:nTest], y[0:nTest], 'b', linewidth=4, alpha=0.75, label='HDNN')
    plt.plot(x_s, y_s, ':g', linewidth=2, label='RK45')
    plt.plot(x_num[0:Nexact], y_num[0:Nexact], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.ylim([0, 0.5])
    plt.xlim([1.5, 2.1])
    
    plt.ylabel('$y$', rotation=0, fontsize=22)
    plt.xlabel('$x$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/gc_xy_trajectory_hd_{w_constr}{kappa0}.pdf', format="pdf", bbox_inches="tight")
    
    # Compute L2 Errors Between Different Methods
    
    # Errors for HDNN vs. exact solution (LSODA)
    dx = x_num - x[0:nTest, 0]
    dpx = px_num - px[0:nTest, 0]
    dy = y_num - y[0:nTest, 0]
    dpy = py_num - py[0:nTest, 0]
    
    # Errors for RK45 vs. exact solution (LSODA)
    dx_s = x_num - x_s
    dpx_s = px_num - px_s
    dy_s = y_num - y_s
    dpy_s = py_num - py_s
    
    # Compute L2 error norms
    nL2hdnn = np.sqrt(dx**2 + dy**2 + dpx**2 + dpy**2)
    nL2rk45 = np.sqrt(dx_s**2 + dy_s**2 + dpx_s**2 + dpy_s**2)
    
    # L2 Error Plot
    plt.figure(figsize=(10, 8))
    plt.plot(t_net[0:nTest], nL2hdnn[0:nTest], 'b', linewidth=2, label='HDNN')  # L2 error for HDNN
    plt.plot(t_s, nL2rk45, ':g', linewidth=2, label='RK45 x10')  # L2 error for RK45
    plt.ylabel('$L_2$ error', rotation=90, fontsize=22)
    plt.xlabel('$t$', rotation=0, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(axis='both', nbins=8)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/gc_L2_error_hd_{w_constr}{kappa0}.pdf', format="pdf", bbox_inches="tight")
