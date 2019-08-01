
#%%
#least square using vertical line - a=(XT.X)^(−1)XT y where a =[a,b].T
import pickle
import numpy as np

def matrix_lstsqr(x, y):
    # Computes the least-squares solution to a linear matrix equation.
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

with open('HW1_data\data1_new.pkl', 'rb') as f:
     data = pickle.load(f)
        
array = np.asarray(data)
#print(array)
x = array[:,0]
y = array[:,1]
slope, intercept = matrix_lstsqr(x, y)
print(intercept)
line_x = [round(min(x)) - 1, round(max(x)) + 1]
line_y = [slope*x_i + intercept for x_i in line_x]
#rint(line_y)
plt.scatter(x,y)
plt.plot(line_x, line_y, color='red')
ftext = 'y = ax + b = {:.3f} + {:.3f}x'        .format(slope, intercept)
plt.figtext(.15,.8, ftext, fontsize=11, ha='left')


#%%



#%%



