import matplotlib.pyplot as plt
import numpy as np
builtin_data = np.load('./results/builtin10/iters3860.npy')
# linear_data = np.load('./results/linear10/iters3860.npy')
pnp = np.load('./results/linear9/iters3860.npy')

builtin = builtin_data
print(builtin[0])
# linear = linear_data[2]
xpts = []
zpts = []
xpt_lin = []
zpt_lin = []
xpts_pnp = []
zpts_pnp = []
drift1 = []
drift2 = []
drift_frame = []
# r_calc_data = linear_data[0]
# c_calc_data = linear_data[1]
# H1_calc = np.array([[1,0,0,0],
#                 [0,1,0,0],
#                 [0,0,1,0],
#                 [0,0,0,1]])


sum = 0
for i in range(len(builtin)):

    xpt = builtin[i][0]
    zpt = builtin[i][1]

    xptn = xpt*np.cos(-8*np.pi/180) - zpt*np.sin(-8*np.pi/180)
    zptn = zpt*np.cos(-8*np.pi/180) + xpt*np.sin(-8*np.pi/180)

    xpts.append(xptn)
    zpts.append(zptn)



    # R_calc = r_calc_data[i]
    # C_calc = c_calc_data[i]
    # # print(R_calc)
    # # print(C_calc)
    # H2_calc = np.hstack((R_calc,C_calc))
    # H2_calc = np.vstack((H2_calc,[0,0,0,1]))

    # # H1 = np.matmul(H1,H2)
    # H1_calc = np.matmul(H1_calc,H2_calc)
    # xpt_calc = H1_calc[0,3]*np.cos(8*np.pi/180) - H1_calc[2,3]*np.sin(8*np.pi/180)
    # zpt_calc = H1_calc[2,3]*np.cos(8*np.pi/180) + H1_calc[0,3]*np.sin(8*np.pi/180)

    # xpt_lin.append(xpt_calc)
    # zpt_lin.append(zpt_calc)

    xpt_pnp = -(pnp[i][0]*np.cos(-8*np.pi/180) - pnp[i][1]*np.sin(-8*np.pi/180))
    zpt_pnp = pnp[i][1]*np.cos(-8*np.pi/180) + pnp[i][0]*np.sin(-8*np.pi/180)

    xpts_pnp.append(xpt_pnp)
    zpts_pnp.append(zpt_pnp)
    
    # # if np.abs((xpt_calc - xpt)) > 9:

    # d1 =  np.sqrt((xpt_calc - xpt)**2 + (zpt_calc - zpt)**2)
    # d2 =  np.sqrt((xpt_pnp - xpt)**2 + (zpt_pnp - zpt)**2)
    
    # drift1.append(d1)
    # drift2.append(d2)
    # drift_frame.append(i)

# print "len of data",len(xpts)
# plt.plot(0,0,'.b')
#avg = sum/len(builtin)
#print('Average Drift:',avg)
plt.plot(xpts,zpts,'.b',label='built-in')
# plt.plot(xpt_lin,zpt_lin,'.r',label ='self')
plt.plot(xpts_pnp,zpts_pnp,'.g',label='self')# with nonlinear pnp')

# plt.plot(drift_frame,drift2,'r',label = 'self with builtin')
# plt.plot(drift_frame,drift1,'g',label = 'self + nonlinear pnp with builtin')

plt.legend()
plt.show()
