import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.stats import norm

plt.ion()

##############################
# Question 1

## load data
#data = np.genfromtxt('my_data.csv',delimiter=',')


X = data[:,0:4]
F = data[:,4:5]
n, d = X.shape

##############################
# Question 2

# define a kernel
kern1 = GPy.kern.Linear(input_dim=d)
kern2 = GPy.kern.RBF(input_dim=d,variance=np.var(F),lengthscale=[.5]*d,ARD=True)
kern = kern1+kern2

print kern
kern['.*lengthscale'] # get more details about length-scales

# define a model
m = GPy.models.gp_regression.GPRegression(X, F, kern)
print m

# optimize the model parameters
m.optimize()
print m
m['.*lengthscale']

# predict at points Xnew
Xnew = np.random.uniform(0,1,(10,d))
mean, var = m.predict(Xnew)

##############################
# Question 3



##############################
# Question 4

def leaveOneOut(m):
    n = m.X.shape[0]
    mean = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        Xloo = np.delete(m.X,i,0)
        Yloo = np.delete(m.Y,i,0)
        mloo = GPy.models.gp_regression.GPRegression(Xloo, Yloo, m.kern.copy())
        mloo[:] = m[:]
        mean[i],var[i] = mloo.predict(m.X[i:i+1,:])
    return(mean,var)


##############################
# Question 5


##############################
# Question 6


##############################
# Question 7
