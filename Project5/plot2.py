import matplotlib.pyplot as plt
import numpy as np
# builtin_data = np.load('./results/builtin10/iters3220.npy')
linear_data = np.load('./results/linear9/iters3860.npy')
# pnp = np.load('./results/linear9/iters3220.npy')

# builtin = builtin_data
# print(builtin[0])
linear = linear_data
xpts = []
zpts = []
xpt_lin = []
zpt_lin = []
xpt_pnp = []
zpt_pnp = []
drift = []
drift_frame = []
# r_calc_data = linear_data[0]
# c_calc_data = linear_data[1]
# H1_calc = np.array([[1,0,0,0],
#                 [0,1,0,0],
#                 [0,0,1,0],
#                 [0,0,0,1]])


sum = 0
for i in range(len(linear)):

    # xpt = builtin[i][0]
    # zpt = builtin[i][1]

    # xptn = xpt*np.cos(-8*np.pi/180) - zpt*np.sin(-8*np.pi/180)
    # zptn = zpt*np.cos(-8*np.pi/180) + xpt*np.sin(-8*np.pi/180)

    # xpts.append(xptn)
    # zpts.append(zptn)



    # R_calc = r_calc_data[i]
    # C_calc = c_calc_data[i]
    # # print(R_calc)
    # # print(C_calc)
    # H2_calc = np.hstack((R_calc,C_calc))
    # H2_calc = np.vstack((H2_calc,[0,0,0,1]))

    # # H1 = np.matmul(H1,H2)
    # H1_calc = np.matmul(H1_calc,H2_calc)
    # xpt_calc = H1_calc[0,3]*np.cos(8*np.pi/180) - zpt*np.sin(8*np.pi/180)
    # zpt_calc = H1_calc[2,3]*np.cos(8*np.pi/180) + xpt*np.sin(8*np.pi/180)

    xpt_lin.append(-(linear[i][0]*np.cos(-8*np.pi/180) - linear[i][1]*np.sin(-8*np.pi/180)))
    zpt_lin.append(linear[i][1]*np.cos(-8*np.pi/180) + linear[i][0]*np.sin(-8*np.pi/180))

    # xpt_pnp.append(-pnp[i][0]*np.cos(8*np.pi/180) - zpt*np.sin(8*np.pi/180))
    # zpt_pnp.append(pnp[i][1]*np.cos(8*np.pi/180) + xpt*np.sin(8*np.pi/180))
    
    # d =  np.sqrt((builtin[i][0] - linear[i][0])**2 + (builtin[i][1] - linear[i][1])**2)
    # drift.append(d)
    # drift_frame.append(i)

# print "len of data",len(xpts)
# plt.plot(0,0,'.b')
#avg = sum/len(builtin)
#print('Average Drift:',avg)
# plt.plot(xpts,zpts,'.b',label='built-in')
plt.plot(xpt_lin,zpt_lin,'.r',label ='self')
# plt.plot(xpt_pnp,zpt_pnp,'.g',label='self with nonlinear pnp')
plt.legend()
#plt.plot(drift_frame,drift,'r')

plt.show()
