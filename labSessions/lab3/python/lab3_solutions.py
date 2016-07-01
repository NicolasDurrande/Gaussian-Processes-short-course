import numpy as np
import pylab as pb
import GPy
pb.ion()

##################################################################
##                           helpers                            ##
##################################################################

## coordinate change

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

## error measures

def Q2(F,mX):
	return(1-sum((F-mX)**2)/sum((F-np.mean(F))**2))

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
# Question 1

# load data
data = np.genfromtxt('lab1_data.csv',delimiter=',')

#X = data[:,0:4]
#F = np.mean(data[:,4:6],axis=1)[:,None] - np.mean(data[:,4:6])
X = np.vstack((data[:,0:4],data[:,0:4]))
F = np.vstack((data[:,4:5],data[:,5:6]))

n, d = X.shape

## choose the new parameterization and rescale for convenience
X =  old2new(X)
limits = np.array([75,115,20,35,22,31,0.65,1.6]).reshape(4,2).T
X = (X-limits[0:1,:])/(limits[1:2,:]-limits[0:1,:])

F = F-np.mean(F)

# look at noise variance
tau2 = np.var(data[:,4]-data[:,5])/2.


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
Xnew = np.random.uniform(0,1,(1000,d))
Xnew = Xnew*(np.max(X,axis=0)-np.min(X,axis=0)) + np.min(X,axis=0)

mean, var = m.predict(Xnew)

print "IMSE =", round(np.mean(var),2)

##############################
# Question 3

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

mlto, vlto = leaveTwoOut(m)

pb.figure()
pb.plot(F,mlto,'kx',mew=2)
pb.plot((-2,2),(-2,2),'k--',linewidth=.75)
pb.xlim((-1.7,1.7)), pb.ylim((-1.7,1.7))
pb.xlabel('real values'),pb.ylabel('LTO predictions')

##############################
# Question 4


##############################
# Question 5

print "Q2 =", round(Q2(F,mloo[:,None])[0],2)


##############################
# Question 6
