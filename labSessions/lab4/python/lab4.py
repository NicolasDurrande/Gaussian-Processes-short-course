import numpy as np
import pylab as pb
import scipy as sp
import GPy
pb.ion()

##################################################################
##                           helpers                            ##
##################################################################

##            coordinate change            ##

def angle(X):
    # input X is ["Wing-length", "Wing-width", "Tail-length", "Arm-length"]
    # output is the angle (in degrees) between the tail and the wing
    return(180/np.pi*np.arccos(-1.*((X[:,3]-2.5)**2-(X[:,2]-2.5)**2-X[:,0]**2)/(2*(X[:,2]-2.5)*X[:,0])))

# mapping to the new space
def old2new(X):
	# input X is ["Wing-length", "Wing-width", "Tail-length", "Arm-length"]
	# output Y is ['wing angle', 'wing area','total length', 'wing_l / tail_l ratio']
	Y = 0*X
	Y[:,0] = angle(X)
	Y[:,1] = X[:,0] * X[:,1]  
	Y[:,2] = X[:,0] + X[:,2] + X[:,3] 
	Y[:,3] = X[:,0] / (X[:,2] - 2.5)
	return(Y)

# mapping back to the original space
def new2old(Y):
	# input Y is ['wing angle', 'wing area','total length', 'wing_l / tail_l ratio']
	# output X is ["Wing-length", "Wing-width", "Tail-length", "Arm-length"]
	X = 0*Y
	f_1 = np.sqrt(1+(1/Y[:,3])**2-2*np.cos(Y[:,0]*np.pi/180)*(1/Y[:,3]))
	X[:,0] = (Y[:,2]-5) / (1 + 1./Y[:,3] + f_1)
	X[:,1] = Y[:,1] / X[:,0]
	X[:,2] = X[:,0] / Y[:,3] + 2.5
	X[:,3] = Y[:,2] - X[:,0] - X[:,2]
	return(X)


##         error measures         ##

def Q2(F,Y):
	# F : vector of target values
	# Y : vector of predicted values
	return(1-sum((F.flatten()-Y.flatten())**2)/sum((F.flatten()-np.mean(F))**2))

def leaveTwoOut(m):
    n = m.X.shape[0]
    mean = np.zeros(n)
    var = np.zeros(n)
    for i in range(n/2):
        Xloo = np.delete(m.X,[i,i+n/2],0)
        Yloo = np.delete(m.Y,[i,i+n/2],0)
        mloo = GPy.models.gp_regression.GPRegression(Xloo, Yloo, m.kern.copy())
        mloo[:] = m[:]
        mean[[i,i+n/2]],var[[i,i+n/2]] = mloo.predict(m.X[[i,i+n/2],:])
    return(mean,var)


##      expected improvement      ##

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
# Question 1

# load data
data = np.genfromtxt('data2015.csv',delimiter=',')
X = data[:,0:4] # initial parameter space
Xnew = old2new(X)

n, d = X.shape
namesOld = ["Wing-length", "Wing-width", "Tail-length", "Arm-length"]
namesNew = ['wing angle', 'wing area','total length', 'wing_l / tail_l ratio']

## basic plots 
pb.figure()
pb.plot(data[:,4],data[:,5],'kx',mew=1.5)
pb.xlabel('experiment 1'), pb.ylabel('experiment 2') 
pb.title('flight duration')

## outputs versus inputs initial coordinates
pb.figure(figsize=(10,10))

pb.subplot(221)
pb.plot(data[:,0],data[:,4],'kx',mew=1.5)
pb.plot(data[:,0],data[:,5],'kx',mew=1.5)
pb.xlabel(namesOld[0]), pb.ylabel('flight duration') 

pb.subplot(222)
pb.plot(data[:,1],data[:,4],'kx',mew=1.5)
pb.plot(data[:,1],data[:,5],'kx',mew=1.5)
pb.xlabel(namesOld[1]), pb.ylabel('flight duration') 

pb.subplot(223)
pb.plot(data[:,2],data[:,4],'kx',mew=1.5)
pb.plot(data[:,2],data[:,5],'kx',mew=1.5)
pb.xlabel(namesOld[2]), pb.ylabel('flight duration') 

pb.subplot(224)
pb.plot(data[:,3],data[:,4],'kx',mew=1.5)
pb.plot(data[:,3],data[:,5],'kx',mew=1.5)
pb.xlabel(namesOld[3]), pb.ylabel('flight duration') 

## outputs versus inputs new coordinates
pb.figure(figsize=(10,10))

pb.subplot(221)
pb.plot(Xnew[:,0:1],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[0]), pb.ylabel('flight duration')

pb.subplot(222)
pb.plot(Xnew[:,1:2],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[1]), pb.ylabel('flight duration')

pb.subplot(223)
pb.plot(Xnew[:,2:3],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[2]), pb.ylabel('flight duration')

pb.subplot(224)
pb.plot(Xnew[:,3:4],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[3]), pb.ylabel('flight duration')

###################################
# remove outliers
F = data[:,4:6].copy()
outl_ind = F[:,0]-F[:,1]>1
F[outl_ind,0] = F[outl_ind,1]

## basic plots 
pb.figure()
pb.plot(F[:,0],F[:,1],'kx',mew=1.5)
pb.xlabel('experiment 1'), pb.ylabel('experiment 2') 
pb.title('flight duration (no outlier)')


###################################
# build GPR model
limits = np.array([75,115,20,35,22,31,0.65,1.6]).reshape(4,2).T
Xnew01 = (Xnew-limits[0:1,:])/(limits[1:2,:]-limits[0:1,:])
F = F-np.mean(F)
F = -F

Xs = np.vstack((Xnew01,Xnew01))
Fs = np.vstack((F[:,0:1],F[:,1:2]))

kern = GPy.kern.Matern52(input_dim=d,variance=0.15,lengthscale=[0.5]*4, ARD=True)
kern['.*lengthscale'].constrain_bounded(0.05,2)
mopt = GPy.models.gp_regression.GPRegression(Xs, Fs, kern)


mopt.optimize_restarts(10)
print mopt
mopt['.*lengthscale']

# test 1 quality of mean
pred_mean , pred_var = leaveTwoOut(mopt)
print "RMSE =", round(np.sqrt(np.mean(np.square(pred_mean-Fs[:,0]))),2)
print "Q2 =", round(Q2(Fs , pred_mean),2)

# test 2 quality of confidence intervals
standardised_error = (pred_mean-Fs[:,0])/np.sqrt(pred_var)
np.mean(standardised_error) # should be around 0
np.std(standardised_error)  # should be around 1

pb.figure()
_ = pb.hist(standardised_error,normed=True)
x = np.linspace(-3,3,100)
pb.plot(x,sp.stats.norm.pdf(x))


##############################
# Question 2

# up to you guys! 
