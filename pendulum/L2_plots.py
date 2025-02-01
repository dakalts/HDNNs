'''

L2 error plot for different values of w_constr


'''
import numpy as np
import matplotlib.pyplot as plt

L2err1=np.loadtxt('data/l2err_0.1.txt')

L2err2=np.loadtxt('data/l2err_1.0.txt')

L2err3=np.loadtxt('data/l2err_10.0.txt')

L2err4=np.loadtxt('data/l2err_100.0.txt')


groupsize=100
group1 = np.array([L2err1[x:x+groupsize] for x in range(0, len(L2err1), groupsize)])
group2 = np.array([L2err2[x:x+groupsize] for x in range(0, len(L2err2), groupsize)])
group3 = np.array([L2err3[x:x+groupsize] for x in range(0, len(L2err3), groupsize)])
group4 = np.array([L2err4[x:x+groupsize] for x in range(0, len(L2err4), groupsize)])
#  calculate the means and stds
mean1 = np.array([group.mean() for group in group1])
std1 = np.array([group.std() for group in group1])
mean2= np.array([group.mean() for group in group2])
std2 = np.array([group.std() for group in group2])
mean3 = np.array([group.mean() for group in group3])
std3 = np.array([group.std() for group in group3])
mean4 = np.array([group.mean() for group in group4])
std4 = np.array([group.std() for group in group4])

xm = np.array([i for i in range(len(mean1))])

plt.figure(figsize=(10,8))
#plt.plot(mean4, 'g-.',alpha=0.6, linewidth=3, label=r'$w_c=0.1$')
plt.plot(mean1,'b', alpha=0.6, linewidth=2, label=r'$w_c=0.1$')
plt.plot(mean2, '--c',alpha=0.6, linewidth=2, label=r'$w_c=1.0$')
plt.plot(mean3, ':r',alpha=0.6, linewidth=3, label=r'$w_c=10.0$')
#plt.fill_between(xm,mean4+std4, mean4-std4, facecolor='green', alpha=0.35)
plt.fill_between(xm,mean1+std1, mean1-std1, facecolor='blue', alpha=0.35)
plt.fill_between(xm,mean2+std2, mean2-std2, facecolor='cyan', alpha=0.35)
plt.fill_between(xm,mean3+std3, mean3-std3, facecolor='red', alpha=0.35)

plt.yscale("log")
plt.ylabel('norm $L_2$ error',rotation =90, fontsize=22)
plt.xlabel('epochs $(x10^2)$',rotation =0, fontsize=22)
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.legend(fontsize=20)

plt.savefig('figs/l2err_comp.pdf',format="pdf", bbox_inches="tight")