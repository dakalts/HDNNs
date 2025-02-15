import numpy as np
import torch
from HDNN_erho_multi_eps import *
import matplotlib.pyplot as plt
from IPython import get_ipython
import time

# Enable interactive plotting
get_ipython().run_line_magic('matplotlib', 'qt')

def pred_solutions(epsilon, P0, N, t_max, model, device, alpha, beta, gamma, w_constr):
    """
    Predicts solutions using a deep learning model and compares them with numerical solvers.
    
    Parameters:
    epsilon : float
        Small parameter controlling perturbation.
    P0 : list
        Initial conditions [t0, x0, y0, px0, py0].
    N : int
        Number of test points.
    t_max : float
        Maximum time for the solution.
    model : PyTorch model
        Neural network model for prediction.
    device : torch.device
        Device to run computations (CPU/GPU).
    alpha, beta, gamma : floats
        System parameters.
    w_constr : float
        Constraint parameter.
    """
    plt.close('all')
    
    # Define test parameters
    nTest = 10*N
    predf=1.0
    t_max_test = predf*t_max
    t0,x0,y0,px0,py0=P0[0],P0[1],P0[2],P0[3],P0[4] 
   
    # Generate test time values
    tTest = torch.linspace(t0,t_max_test,nTest).to(device) 
    tTest = tTest.reshape(-1,1);
    tTest.requires_grad=True
    t_net = tTest.detach().cpu().numpy()
    
    # Compute predicted solutions using neural network
    x, y, px, py= parametricSolutions(tTest,model,P0,torch.tensor([[epsilon]]).float().to(device),gamma)
    x = x.data.cpu().numpy()
    y=y.data.cpu().numpy()
    px = px.data.cpu().numpy()
    py=py.data.cpu().numpy()
    
    # Compute energy of predicted solutions
    E = energy(x, y, px, py, alpha, beta, epsilon)
    
    # Solve using LSODA
    t_num = np.linspace(t0, t_max_test, nTest)
    x_num, y_num, px_num, py_num = LSODA(N, t_num, x0, y0, px0, py0, alpha, beta, epsilon)
    x_un, y_un, px_un, py_un = LSODA_un(N, t_num, x0, y0, px0, py0, alpha, beta, epsilon)
    E_num = energy(x_num, y_num, px_num, py_num, alpha, beta, epsilon)
    
    # Solve using RK45 method
    startT_RK45 = time.time()
    E_s, x_s, y_s, px_s, py_s, t_s = RK45(nTest-1, x0, y0, px0, py0, t0, t_max_test, alpha, beta, epsilon)
    runTimeRK45 = (time.time() - startT_RK45) / 60
    print('RK45 runtime is', runTimeRK45)
    
    # Define number of test points for plots
    nTestp = int(nTest * predf)
    
    # Plot x and y timeseries
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_net[:nTestp], x[:nTestp], 'b', alpha=0.85, linewidth=2, label='HDNN')
    plt.plot(t_s[:nTestp], x_s[:nTestp], ':g', alpha=0.85, linewidth=3, label='RK45')
    plt.plot(t_num[:nTestp], x_num[:nTestp], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.ylabel('$x$', fontsize=22)
    plt.xlabel('$t$', fontsize=22)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_net[:nTestp], y[:nTestp], 'b', alpha=0.85, linewidth=2, label='HDNN')
    plt.plot(t_s[:nTestp], y_s[:nTestp], ':g', alpha=0.85, linewidth=3, label='RK45')
    plt.plot(t_num[:nTestp], y_num[:nTestp], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.ylabel('$y$', fontsize=22)
    plt.xlabel('$t$', fontsize=22)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'figs/erho_xy_timeseries_hd_{w_constr}{epsilon}.pdf', format='pdf', bbox_inches='tight')
    
    # Plot px and py timeseries
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(t_net[0:nTestp], px[0:nTestp], 'b',alpha=0.85, linewidth=2, label='HDNN')
  #  plt.plot(t_net[nTestp:nTest], px[nTestp:nTest],'-.c', linewidth=1, label='prediction')
    plt.plot(t_s[0:nTestp],px_s[0:nTestp],':g',alpha=0.85,linewidth=3, label='RK45')
    plt.plot(t_num[0:nTestp], px_num[0:nTestp],'--r',alpha=0.85,linewidth=2, label='LSODA')
    #plt.plot(t_s,x_s,'-.r',linewidth=lwdth, label='RK45 x 10 points'); 
    plt.ylabel('$p_{x}$',rotation = 0 , fontsize=22)
    plt.xlabel('$t$',fontsize=22)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.locator_params(axis='both', nbins=8) 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
    
    
    plt.subplot(2,1,2)
    plt.plot(t_net[0:nTestp], py[0:nTestp],'b', alpha=0.85,linewidth=2, label='HDNN')
   # plt.plot(t_net[nTestp:nTest], py[nTestp:nTest],'-.c', linewidth=1, label='prediction')
    plt.plot(t_s[0:nTestp], py_s[0:nTestp],':g',alpha=0.85,linewidth=3, label='RK45')
    plt.plot(t_num[0:nTestp], py_num[0:nTestp],'--r',alpha=0.85,linewidth=2, label='LSODA'); 
    plt.ylabel('$p_{y}$', rotation = 0 , fontsize=20)
    plt.xlabel('$t$',fontsize=20)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
    
    plt.tight_layout()
    plt.savefig('figs/erho_p_timeseries_hd_' + str(w_constr) + str(epsilon) + '.pdf',format="pdf", bbox_inches="tight")

    
    # Compute energy deviation
    E0 = 0.5 * (px0**2 + py0**2) + 0.5 * alpha * (x0**2) + 0.5 * beta * (y0**2)
    dE = (E - E0) / E0
    dE_s = (E_s - E0) / E0
    dE_num = (E_num - E0) / E0
    
    # Compute means and standard deviations of energy deviations
    groupsize = 100
    mean_dE = np.array([0.0] + [np.mean(dE[i:i+groupsize]) for i in range(0, len(dE)-groupsize, groupsize)])
    mean_dE_s = np.array([0.0] + [np.mean(dE_s[i:i+groupsize]) for i in range(0, len(dE_s)-groupsize, groupsize)])
    mean_dE_num = np.array([0.0] + [np.mean(dE_num[i:i+groupsize]) for i in range(0, len(dE_num)-groupsize, groupsize)])
    
    # Time array for plotting
    tm = np.linspace(0, t_max_test, len(mean_dE))
    
    # Plot energy deviation
    plt.figure(figsize=(10, 6))
    plt.plot(tm, mean_dE, 'b', linewidth=2, label='HDNN')
    plt.plot(tm, mean_dE_s, ':g', linewidth=3, label='RK45')
    plt.plot(tm, mean_dE_num, '--r', linewidth=2, label='LSODA')
    plt.fill_between(tm, mean_dE_s + np.std(dE_s), mean_dE_s - np.std(dE_s), facecolor='green', alpha=0.35)
    plt.fill_between(tm, mean_dE + np.std(dE), mean_dE - np.std(dE), facecolor='blue', alpha=0.35)
    plt.fill_between(tm, mean_dE_num + np.std(dE_num), mean_dE_num - np.std(dE_num), facecolor='red', alpha=0.35)
    plt.ylabel('$\Delta E / E_0$', fontsize=22)
    plt.xlabel('$t$', fontsize=22)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/erho_energy_hd_{w_constr}{epsilon}.pdf', format='pdf', bbox_inches='tight')
    
    
    # Compute semi-major and semi-minor axes
    a = np.sqrt(x0**2 + y0**2 / (1 - epsilon**2))  # Semi-major axis
    b = np.sqrt(a**2 * (1 - epsilon**2))  # Semi-minor axis
    
    
    # Generate angles for plotting the constraint ellipse
    t = np.linspace(0, 2 * np.pi, 100)
    
    # Define number of exact test points
    nExact = int(0.066 * nTest)
    
    ### Plot x-y trajectory ###
    plt.figure(figsize=(10, 10))
    plt.plot(x_un[:nExact], y_un[:nExact], '-.c', alpha=0.75, linewidth=3, label='Unrestricted')
    plt.plot(a * np.cos(t), b * np.sin(t), '--k', label='Constraint')
    plt.plot(x_s[:nTest], y_s[:nTest], ':g', linewidth=3, alpha=0.85, label='RK45')
    plt.plot(x_num[:nTest], y_num[:nTest], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.plot(x[:int(nTest / predf)], y[:int(nTest / predf)], 'b', linewidth=4, alpha=0.75, label='HDNN')
    
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', rotation=0, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/erho_xy_trajectory_hd_{w_constr}{epsilon}.pdf', format="pdf", bbox_inches="tight")
    
    ### Plot phase-space trajectory (x vs. px) ###
    plt.figure(figsize=(10, 10))
    plt.plot(x_un[:nExact], px_un[:nExact], '-.c', alpha=0.75, linewidth=3, label='Unrestricted')
    plt.plot(x_s[:nTest], px_s[:nTest], ':g', linewidth=3, alpha=0.85, label='RK45')
    plt.plot(x_num[:nTest], px_num[:nTest], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.plot(x[:int(nTest / predf)], px[:int(nTest / predf)], 'b', linewidth=3, alpha=0.75, label='HDNN')
    
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$p_x$', rotation=0, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/erho_ps_trajectory_hd_{w_constr}{epsilon}.pdf', format="pdf", bbox_inches="tight")
    
    ### Plot px vs. py trajectory ###
    plt.figure(figsize=(10, 10))
    plt.plot(px_un[:nExact], py_un[:nExact], '-.c', alpha=0.75, linewidth=3, label='Unrestricted')
    plt.plot(px_s[:nTest], py_s[:nTest], ':g', linewidth=3, alpha=0.85, label='RK45')
    plt.plot(px_num[:nTest], py_num[:nTest], '--r', alpha=0.85, linewidth=2, label='LSODA')
    plt.plot(px[:int(nTest / predf)], py[:int(nTest / predf)], 'b', linewidth=3, alpha=0.75, label='HDNN')
    
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', rotation=0, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figs/erho_pxpy_trajectory_hd_{w_constr}{epsilon}.pdf', format="pdf", bbox_inches="tight")
    
    ### Compute constraint violations ###
    ls = x0**2 + y0**2 / (1 - epsilon**2)
    Phi = abs(x**2 + y**2 / (1 - epsilon**2) - ls)
    Psi = abs(x * px + y * py / (1 - epsilon**2) - (x0 * px0 + y0 * py0 / (1 - epsilon**2)))
    
    Phi_s = abs(x_s**2 + y_s**2 / (1 - epsilon**2) - ls)
    Psi_s = abs(x_s * px_s + y_s * py_s / (1 - epsilon**2) - (x0 * px0 + y0 * py0 / (1 - epsilon**2)))
    
    Phi_num = abs(x_num**2 + y_num**2 / (1 - epsilon**2) - ls)
    Psi_num = abs(x_num * px_num + y_num * py_num / (1 - epsilon**2) - (x0 * px0 + y0 * py0 / (1 - epsilon**2)))
    
    ### Plot constraint violations ###
    plt.figure(figsize=(10, 8))
    
    # Phi constraint plot
    plt.subplot(2, 1, 1)
    plt.plot(t_net[:nTestp], Phi[:nTestp], 'b', alpha=0.85, linewidth=2, label='HDNN')
    plt.plot(t_s[:nTestp], Phi_s[:nTestp], ':g', alpha=0.85, linewidth=3, label='RK45')
    plt.plot(t_num[:nTestp], Phi_num[:nTestp], '--r', alpha=0.85, linewidth=2, label='LSODA')
    
    plt.ylabel('$|\Phi|$', rotation=0, fontsize=22)
    plt.xlabel('$t$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    
    # Psi constraint plot
    plt.subplot(2, 1, 2)
    plt.plot(t_net[:nTestp], Psi[:nTestp], 'b', alpha=0.85, linewidth=2, label='HDNN')
    plt.plot(t_s[:nTestp], Psi_s[:nTestp], ':g', alpha=0.85, linewidth=3, label='RK45')
    plt.plot(t_num[:nTestp], Psi_num[:nTestp], '--r', alpha=0.85, linewidth=2, label='LSODA')
    
    plt.ylabel('$|\Psi|$', rotation=0, fontsize=22)
    plt.xlabel('$t$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'figs/erho_constrs_hd_{w_constr}{epsilon}.pdf', format="pdf", bbox_inches="tight")
