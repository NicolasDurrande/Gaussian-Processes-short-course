import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.stats import norm

plt.ion()

##############################
# Question 1

# load data
data = np.genfromtxt('lab1_data.csv',delimiter=',')

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
Xnew = np.random.uniform(0,1,(1000,d))
mean, var = m.predict(Xnew)

print "IMSE =", round(np.mean(var),2)

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

mloo, vloo = leaveOneOut(m)

plt.figure()
plt.plot(F,mloo,'kx',mew=2)
plt.plot((0,200),(0,200),'k--',linewidth=.75)
plt.xlabel('real values'),plt.ylabel('LOO predictions')

##############################
# Question 5

def Q2(F,Y):
	# F : vector of target values
	# Y : vector of predicted values
    F = F.flatten()
    Y = Y.flatten()
    q2 = 1-sum((F-Y)**2)/sum((F-np.mean(F))**2)
	return(q2)

print "Q2 =", round(Q2(F,mloo),2)

##############################
# Question 6

# standardised residuals
std_res = (mloo-F.flatten()) / np.sqrt(vloo)

x = np.linspace(-3,3,50)

plt.figure()
plt.hist(std_res,bins=5,normed=True)
plt.plot(x,norm.pdf(x))

##############################
# Question 7
