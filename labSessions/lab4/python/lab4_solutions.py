import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import GPy
plt.ion()

##################################################################
##                           helpers                            ##
##################################################################

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

def Q2(F,Y):
	# F : vector of target values
	# Y : vector of predicted values
    F = F.flatten()
    Y = Y.flatten()
    q2 = 1-sum((F-Y)**2)/sum((F-np.mean(F))**2)
	return(q2)

def EI(x,m): # expected improvement for deterministic model
	# x : matrix of input points
	# m : GPy GPR model
	mean, var = m.predict(x)
	var[var<0] = 0
	u = (np.min(m.Y) - mean)/np.sqrt(var)
	ei = np.sqrt(var) * (u * sp.stats.norm.cdf(u) + sp.stats.norm.pdf(u))
	ei[np.isnan(ei)] = 0
	return(ei)

##################################################################
##                          Questions                           ##
##################################################################

##############################
# Question 0

# load data
data = np.genfromtxt('lab1_data.csv',delimiter=',')

X = data[:,0:4]
F = -data[:,4:5]
n, d = X.shape

# define a kernel
kern1 = GPy.kern.Bias(d,variance=np.var(F)/2)
kern2 = GPy.kern.RBF(d,variance=np.var(F)/2,lengthscale=[.5]*d,ARD=True)
kern = kern1+kern2
#kern['.*lengthscale'].constrain_bounded(0.2,20)

# define and optimize the model
m = GPy.models.gp_regression.GPRegression(X, F, kern)
m.optimize_restarts(10)

print m
m['.*lengthscale']

# test the model mean
mloo, vloo = leaveOneOut(m)
print "MSE =", round(np.mean((mloo-F[:,0])**2),2)
print "Q2 =", round(Q2(F,mloo),2)

# test 2 quality of confidence intervals
standardised_error = (mloo-F[:,0])/np.sqrt(vloo)
np.mean(standardised_error) # should be around 0
np.std(standardised_error)  # should be around 1

plt.figure()
plt.hist(standardised_error,normed=True)
x = np.linspace(-3,3,100)
plt.plot(x,sp.stats.norm.pdf(x))

##############################
# Question 1

# The important thing not to forget is to change F for -F (as we did in line 52) since most optimizers perform minimisation by default.

##############################
# Question 2

def EIn(x,m): # expected improvement for noisy model
	# x : matrix of input points
	# m : GPy GPR model
	mX, vX = m.predict(m.X)
	mn = GPy.models.gp_regression.GPRegression(m.X, mX, m.kern)
	mn['.*nois'] = 0.0
	return(EI(x,mn))

##############################
# Question 3

x = np.random.uniform(0,1,(10000,4))
ei = EIn(x,m)
np.max(ei)

xopt = x[np.argmax(ei),:]

## Plot location of the new point
plt.figure(figsize=(10,10))
for i in range(d):
    plt.subplot(2,2,i+1)
    plt.plot(X[:,i:i+1],-F,'kx',mew=1.5)
    plt.axvline(xopt[i],color='g')
    plt.xlabel("x{}".format(i))
    plt.ylabel('shot distance')

# For xopt, the simulator returns 180, so we have significanly improved the distance!  
