import numpy as np
import pylab as pb
import scipy as sp
import GPy
pb.ion()

##############################
# functions

def Q2(F,mX):
	return(1-sum((F-mX)**2)/sum((F-np.mean(F))**2))

# This functions converts the original parameters into the new one:
# Wl,Ww,Tl,Al -> Wl, Wp (Wing area), Hl (Total helicopter length), Theta (Wing angle)
def OP2NP(X):
    theta = np.arccos( - (X[:,3]**2 - X[:,2]**2 - X[:,0]**2 ) / (2* X[:,2] * X[:,0] ) )
    #print np.shape(theta)
    wingSurface = X[:,0] * X[:,1] 
    #print np.shape(wingSurface)
    hLength = X[:,2] + X[:,3]
    #print np.shape(hLength)
    XN = np.vstack((X[:,0], wingSurface, hLength, theta))
    XN=XN.T
    return(XN)

def leaveOneOut(m):
    n = m.X.shape[0]
    mean = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        Xloo = np.delete(m.X,i,0)
        Yloo = np.delete(m.Y,i,0)
        mloo = GPy.models.gp_regression.GPRegression(Xloo, Yloo, m.kern.copy())
        mloo[:] = m[:]
        mean[i],var[i] = mloo.predict(X[i:i+1,:])
    return(mean.T,var)

def EI(x,m):
	mean, var = m.predict(x)
	var[var<0] = 0
	u = (np.min(m.Y) - mean)/np.sqrt(var)
	ei = np.sqrt(var) * (u * sp.stats.norm.cdf(u) + sp.stats.norm.pdf(u))
	ei[np.isnan(ei)] = 0
	return(ei)

def EIn(x,m):
	mX, vX = m.predict(m.X)
	mn = GPy.models.gp_regression.GPRegression(X, mX, m.kern)
	mn['.*nois'] = 0.0
	mean, var = mn.predict(x)
	var[var<0] = 0
	u = (np.min(mX) - mean)/(np.sqrt(var))
	ei = np.sqrt(var) * (u * sp.stats.norm.cdf(u) + sp.stats.norm.pdf(u))
	ei[np.isnan(ei)] = 0
	return(ei)


##############################
# load data
data = np.genfromtxt('ourData.csv',delimiter=',')
X = data[:,0:4] # initial parameter space
d = X.shape[1]

# remove outliers
F = data[:,4:6]
G = F.copy()
G[F[:,0]-F[:,1]>1,0] = F[F[:,0]-F[:,1]>1,1]
F = np.mean(G,axis=1)[:,None] - np.mean(data[:,4:6])

## basic plots 
for i in range(4):
	pb.figure()
	pb.plot(X[:,i],F,'kx',mew=1.5)
	pb.title('time vs variable %i'%(i+1))

## basic plots 
pb.plot(data[:,4],data[:,5],'kx',mew=1.5)
pb.title('experiment 1 vs experiment 2')
pb.xlim((G.min(),G.max()))
pb.ylim((G.min(),G.max()))

###################################
# build GPR model

kernopt = GPy.kern.RBF(input_dim=d,variance=0.15,lengthscale=[0.6,1.5,10,0.1], ARD=True)
kernopt = GPy.kern.RBF(input_dim=d,variance=0.22,lengthscale=[1.6,3.5,5.5,1.5], ARD=True)
mopt = GPy.models.gp_regression.GPRegression(X, F, kernopt)
mopt['.*noise'].fix(0.1)  # fix noise variance

mopt.optimize_restarts(10)
print mopt
mopt['.*lengthscale']

# test 1 quality of mean
mn , v = leaveOneOut(mopt)
Q2(F , mn[:,None])


# test 2 quality of confidence intervals
stan_res = (mn-F[:,0])/np.sqrt(v)
pb.figure()
_ = pb.hist(stan_res,normed=True)
x = np.linspace(-3,3,100)
pb.plot(x,sp.stats.norm.pdf(x))

x = np.random.uniform(np.min(X,0),np.max(X,0),(10000,4))
ei = EIn(x,mopt)
np.max(ei)

