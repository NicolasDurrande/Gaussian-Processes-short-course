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
data = TODO      # load here your data

X = data[:,0:4]
F = data[:,4:5]
n, d = X.shape

m = TODO         # retreive here your best model from lab 3

# test 1 : the model mean
mloo, vloo = leaveOneOut(m)
print "MSE =", round(np.mean((mloo-F[:,0])**2),2)
print "Q2 =", round(Q2(F,mloo),2)

# test 2 : the model variance
standardised_error = (mloo-F[:,0])/np.sqrt(vloo)
np.mean(standardised_error) # should be around 0
np.std(standardised_error)  # should be around 1

plt.figure()
plt.hist(standardised_error,normed=True)
x = np.linspace(-3,3,100)
plt.plot(x,sp.stats.norm.pdf(x))

##############################
# Question 1

##############################
# Question 2

##############################
# Question 3

##############################
# Question 4
