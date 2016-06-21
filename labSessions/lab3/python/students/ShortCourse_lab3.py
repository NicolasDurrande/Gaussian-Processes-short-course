import numpy as np
import pylab as pb
import GPy
pb.ion()

##################################################################
##                           helpers                            ##
##################################################################

##############################
# functions
def Q2(F,mX):
	return(1-sum((F-mX)**2)/sum((F-np.mean(F))**2))


##############################
# GPy example

# load data
data = np.genfromtxt('lab1_data.csv',delimiter=',')
X = data[:,0:4]
F = np.mean(data[:,4:6],axis=1)[:,None] - np.mean(data[:,4:6])
d = X.shape[1]

# define noise variance
tau2 = np.var(data[:,4]-data[:,5])/2

# define a kernel
kern = GPy.kern.Matern32(input_dim=d,variance=np.var(F),lengthscale=[5]*d,ARD=True)
print kern
kern['lengthscale']

# define a model
m = GPy.models.gp_regression.GPRegression(X, F, kern)
m['.*noise'].fix(tau2)		# fix the noise variance
print m

# optimize the model parameters
m.optimize()
print m
m['.*lengthscale']

# predict at points X
mX,varX = m.predict(X)
Q2(F,mX)

# predict at points Xnew
Xnew = np.random.uniform(0,1,(1000,d))
Xnew = Xnew*(np.max(X,axis=0)-np.min(X,axis=0)) + np.min(X,axis=0)

mean, var = m.predict(Xnew)

##################################################################
##                             TODO                             ##
##################################################################

def leaveOneOut(m):
	n = m.X.shape[0]
	mean = np.zeros(n)
	var = np.zeros(n)
	for i in range(n):
		Xloo = np.delete(m.X,i,0)
		Yloo = np.delete(m.Y,i,0)
		mloo = GPy.models.gp_regression.GPRegression(Xloo, Yloo, kern.copy())
		mloo[:] = m[:]
		mean[i],var[i] = mloo.predict(X[i:i+1,:])
	return(mean,var)

leaveOneOut(m)


arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

np.delete(arr, 0, 1)
