# Import necessary libraries
import numpy as np
import torch
from HDNN_pend_multi_m import *  # Custom module for HDNN solutions
from pend_helpers import RK45, LSODA, energy, saveData  # Custom helper functions
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')  # Set matplotlib to use Qt for plotting
import time

# Define a function to predict and plot solutions for the pendulum system
def pred_solutions(m0, P0, N, t_max, model, device, gamma, n_train, Tpend, w_constr, dt):
    plt.close('all')  # Close all existing plots

    # Extract initial conditions and constants from P0
    t0, px0, py0, theta0, g = P0[0], P0[3], P0[4], P0[5], P0[6]
    
    # Convert theta0 to Cartesian coordinates
    x0 = np.cos(theta0)
    y0 = np.sin(theta0)

    # Define the number of test points and the maximum time for testing
    nTest = 10 * N 
    predf = 1.0
    t_max_test = predf * t_max
    tTest = torch.linspace(t0, t_max_test, nTest).to(device)  # Create a time tensor
    
    tTest = tTest.reshape(-1, 1)  # Reshape for compatibility
    tTest.requires_grad = True  # Enable gradient computation
    t_net = tTest.detach().cpu().numpy()  # Convert to numpy array for plotting
    
    # Compute parametric solutions using the HDNN model
    x, y, px, py = parametricSolutions(tTest, model, torch.tensor([[m0]]).float().to(device), P0, gamma)
    x = x.data.cpu().numpy()  # Convert to numpy array
    y = y.data.cpu().numpy()
    px = px.data.cpu().numpy()
    py = py.data.cpu().numpy()
    
    # Compute energy for the HDNN solutions
    E = energy(x, y, px, py, m0, g)
    E0 = 0.5 * (px0**2 + py0**2) / m0 + m0 * g * (1 + y0)  # Initial energy
    
    # Solve the system using LSODA (numerical solver)
    t_num = np.linspace(t0, t_max_test, nTest)
    x_num, y_num, px_num, py_num = LSODA(N, t_num, x0, y0, px0, py0, m0, g)
    E_num = energy(x_num, y_num, px_num, py_num, m0, g)
    
    # Solve the system using RK45 
    startT_RK45 = time.time()
    Ns = n_train - 1
    E_s, x_s, y_s, px_s, py_s, t_s = RK45(nTest - 1, x0, y0, px0, py0, t0, t_max_test, m0, g)
    endT_RK45 = time.time()
    runTimeRK45 = endT_RK45 - startT_RK45
    print('RK45 runtime is ', runTimeRK45 / 60)  # Print runtime in minutes
    
    # Plotting the results
    nTestp = int(nTest / predf)
    
    # Plot x vs time
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_net[int(nTestp / 2):nTestp] / Tpend, x[int(nTestp / 2):nTestp], 'b', alpha=0.85, linewidth=2, label='HDNN')
    plt.plot(t_s[int(nTestp / 2):nTest] / Tpend, x_s[int(nTestp / 2):nTest], ':g', alpha=0.85, linewidth=3, label='RK45')
    plt.plot(t_num[int(nTestp / 2):nTest] / Tpend, x_num[int(nTestp / 2):nTest], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.ylabel('$x$', rotation=0, fontsize=22)
    plt.xlabel('$t/T$', fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='y', nbins=6) 
    plt.legend(bbox_to_anchor=(1, 0.5), fontsize=20)
    
    # Plot px vs time
    plt.subplot(2, 1, 2)
    plt.plot(t_net[int(nTestp / 2):nTestp] / Tpend, px[int(nTestp / 2):nTestp], 'b', alpha=0.85, linewidth=2, label='HDNN')
    plt.plot(t_s[int(nTestp / 2):nTest] / Tpend, px_s[int(nTestp / 2):nTest], ':g', alpha=0.85, linewidth=3, label='RK45')
    plt.plot(t_num[int(nTestp / 2):nTest] / Tpend, px_num[int(nTestp / 2):nTest], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.ylabel('$p_x$', rotation=0, fontsize=22)
    plt.xlabel('$t/T$', fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='y', nbins=6) 
    plt.legend(bbox_to_anchor=(1, 0.5), fontsize=20)
    
    plt.tight_layout()
    plt.savefig('figs/pend_xpx_timeseries_hd_' + str(w_constr) + str(m0) + '.pdf', format="pdf", bbox_inches="tight")
    
    # Energy error calculation and plotting
    dE = (E - E0) / E0
    dE_s = (E_s - E0) / E0
    dE_num = (E_num - E0) / E0
     
    groupsize = 100
    groups_dE = np.array([dE[x:x + groupsize] for x in range(0, len(dE), groupsize)])
    groups_dE_s = np.array([dE_s[x:x + groupsize] for x in range(0, len(dE_s), groupsize)])
    groups_dE_num = np.array([dE_num[x:x + groupsize] for x in range(0, len(dE_num), groupsize)])
     
    mean_dE = np.array([0.0] + [group.mean() for group in groups_dE[0:-1]])     
    std_dE = np.array([group.std() for group in groups_dE])
    mean_dE_s = np.array([group.mean() for group in groups_dE_s])
    std_dE_s = np.array([group.std() for group in groups_dE_s])
    mean_dE_num = np.array([group.mean() for group in groups_dE_num])
    std_dE_num = np.array([group.std() for group in groups_dE_num])
     
    NdE = len(mean_dE)
    NdEp = int(len(mean_dE) / predf)
    tm = np.array([i for i in range(len(mean_dE))]) * t_max_test / nTest
     
    plt.figure(figsize=(10, 6))
    plt.plot(tm[0:NdEp] / Tpend, mean_dE[0:NdEp], 'b', linewidth=2, label='HDNN')
    plt.plot(tm / Tpend, np.zeros_like(mean_dE), '--k', linewidth=2, label='exact')
    plt.plot(tm / Tpend, mean_dE_s, ':g', linewidth=3, label='RK45')
     
    plt.fill_between(tm / Tpend, mean_dE_s[0:NdE] + std_dE_s[0:NdE], mean_dE_s[0:NdE] - std_dE_s[0:NdE], facecolor='green', alpha=0.35)
    plt.fill_between(tm[0:NdEp] / Tpend, mean_dE[0:NdEp] + std_dE[0:NdEp], mean_dE[0:NdEp] - std_dE[0:NdEp], facecolor='blue', alpha=0.35)
        
    plt.ylabel('$\Delta E/E_0$  ', rotation=0, fontsize=22)
    plt.xlabel('$t/T$ ($x10^2$)', rotation=0, fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='both', nbins=8) 
    plt.legend(bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig('figs/pend_energy_hd_' + str(w_constr) + str(m0) + '.pdf', format="pdf", bbox_inches="tight")
    
    # Constraints calculations and plotting
    ErrPhi = abs(x**2 + y**2 - x0**2 - y0**2)
    ErrPhi_s = abs(x_s**2 + y_s**2 - x0**2 - y0**2)
    ErrPhi_num = abs(x_num**2 + y_num**2 - x0**2 - y0**2)
    
    ErrPsi = abs(x * px + y * py)
    ErrPsi_s = abs(x_s * px_s + y_s * py_s)
    ErrPsi_num = abs(x_num * px_num + y_num * py_num)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t_net / Tpend, ErrPhi, 'b', linewidth=2, alpha=0.75, label='HDNN')
    plt.plot(t_s / Tpend, ErrPhi_s, ':g', linewidth=2, alpha=0.85, label='RK45') 
    plt.plot(t_num / Tpend, ErrPhi_num, '--k', alpha=0.85, linewidth=2, label='exact')
    plt.ylabel(r'$|\Phi|    $     ', rotation=0, fontsize=22)
    plt.xlabel(r't/T', fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='y', nbins=6) 
    plt.legend(bbox_to_anchor=(0.75, 0.4), fontsize=20)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_net / Tpend, ErrPsi, 'b', linewidth=3, alpha=0.75, label='HDNN')
    plt.plot(t_s / Tpend, ErrPsi_s, ':g', linewidth=2, alpha=0.85, label='RK45') 
    plt.plot(t_num / Tpend, ErrPsi_num, '--k', alpha=0.85, linewidth=2, label='exact')
    plt.ylabel(r'$|\Psi|    $     ', rotation=0, fontsize=22)
    plt.xlabel(r't/T', fontsize=22)
    plt.locator_params(axis='y', nbins=6) 
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.legend(bbox_to_anchor=(1.0, 0.5), fontsize=20)
    
    plt.tight_layout()
    plt.savefig('figs/pend_L2constr_hd_' + str(w_constr) + str(m0) + '.pdf', format="pdf", bbox_inches="tight")
    
    # Pendulum trajectories in x-y and px-py space
    Nexact = int(8.2 * 0.5 * Tpend / dt)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x_s, y_s, ':g', linewidth=2, alpha=0.85, label='RK45') 
    plt.plot(x[0:int(nTest / predf)], y[0:int(nTest / predf)], 'b', linewidth=4, alpha=0.75, label='HDNN')
    plt.plot(x_num[0:Nexact], y_num[0:Nexact], '--r', alpha=0.85, linewidth=2, label='exact')
    plt.ylabel('y', rotation=0, fontsize=22)
    plt.xlabel('x', fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='both', nbins=6) 
    plt.gca().set_aspect('equal')
    plt.legend(bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    plt.plot(px_s, py_s, ':g', linewidth=2, alpha=0.85, label='RK45') 
    plt.plot(px[0:int(nTest / predf)], py[0:int(nTest / predf)], 'b', linewidth=4, alpha=0.75, label='HDNN')
    plt.plot(px_num[0:2 * Nexact], py_num[0:2 * Nexact], '--r', alpha=0.85, linewidth=2, label='exact')
    plt.ylabel('$p_y$', rotation=0, fontsize=22)
    plt.xlabel('$p_x$', fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='both', nbins=6) 
    plt.gca().set_aspect('equal')
    plt.legend(bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig('figs/pend_xy-pxpy_trajectories_hd_' + str(w_constr) + str(m0) + '.pdf', format="pdf", bbox_inches="tight")
    
    # Pendulum phase space trajectory
    plt.figure(figsize=(10, 10))
    plt.plot(x[0:int(nTest / predf)], px[0:int(nTest / predf)], 'b', linewidth=4, alpha=0.75, label='HDNN')
    plt.plot(x_s, px_s, ':g', linewidth=2, alpha=0.85, label='RK45') 
    plt.plot(x_num[0:2 * Nexact], px_num[0:2 * Nexact], '--r', alpha=0.85, linewidth=2, label='exact')
    plt.ylabel('$p_x$', rotation=0, fontsize=22)
    plt.xlabel('$x$', fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='both', nbins=6) 
    plt.legend(bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig('figs/pend_ps_trajectory_hd_' + str(w_constr) + str(m0) + '.pdf', format="pdf", bbox_inches="tight")
    
    # Calculate errors for HDNN and RK45 solutions
    dx = x_num - x[:, 0]       
    dpx = px_num - px[:, 0]
    dy = y_num - y[:, 0]       
    dpy = py_num - py[:, 0]
    
    x_numN, y_numN, px_numN, py_numN = LSODA(Ns, t_s, x0, y0, px0, py0, m0, g) 
    dx_s = x_numN - x_s        
    dpx_s = px_numN - px_s
    dy_s = y_numN - y_s        
    dpy_s = py_numN - py_s
    
    nL2hdnn = np.sqrt(dx**2 + dy**2 + dpx**2 + dpy**2)
    nL2rk45 = np.sqrt(dx_s**2 + dy_s**2 + dpx_s**2 + dpy_s**2)
    
    # Plot L2 errors
    plt.figure(figsize=(10, 8))
    plt.plot(t_net[0:int(nTest / predf)] / Tpend, nL2hdnn[0:int(nTest / predf)], 'b', linewidth=2, label='HDNN')
    plt.plot(t_s / Tpend, nL2rk45, ':g', linewidth=2, label='RK45')
    plt.ylabel('$L_2$ error', rotation=90, fontsize=22)
    plt.xlabel('$t/T$', rotation=0, fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.legend(fontsize=20)
    
    plt.tight_layout()
    plt.savefig('figs/pend_trajectories_error_hd_' + str(w_constr) + str(m0) + '.pdf', format="pdf", bbox_inches="tight")