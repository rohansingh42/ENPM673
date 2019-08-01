import numpy as np
import matplotlib.pyplot as plt
import pickle
f = open('data1_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
plt.figure(1)
plt.subplot(131)
plt.plot(X,Y,'ro')
plt.axis([-150,150,-100,100])
f = open('data2_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
plt.figure(1)
plt.subplot(132)
plt.plot(X,Y,'ro')
plt.axis([-150,150,-100,100])
f = open('data3_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
plt.figure(1)
plt.subplot(133)
plt.plot(X,Y,'ro')
plt.axis([-150,150,-100,100])
plt.show()