import numpy as np
import pylab as pb
import scipy.stats as stats
pb.ion()

#########################
## load the data
# first 4 columns are input: Wing-length, Wing-width, Tail-length, Arm-length
# last 2 columns are outputs in seconds (flight time for two trials)
data = np.genfromtxt('lab1_data.csv',delimiter=',')

X = data[:,0:4]
XS = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)) # inputs rescaled on [0,1]

names = ["Wing-length", "Wing-width", "Tail-length", "Arm-length"]

F = np.mean(data[:,4:6],axis=1)[:,None]

#########################
## Question 1

## outputs versus inputs
pb.figure(figsize=(10,10))

pb.subplot(221)
pb.plot(data[:,0],data[:,4],'kx',mew=1.5)
pb.plot(data[:,0],data[:,5],'kx',mew=1.5)
pb.ylabel('falling time'), pb.xlabel(names[0])

pb.subplot(222)
pb.plot(data[:,1],data[:,4],'kx',mew=1.5)
pb.plot(data[:,1],data[:,5],'kx',mew=1.5)
pb.ylabel('falling time'), pb.xlabel(names[1])

pb.subplot(223)
pb.plot(data[:,2],data[:,4],'kx',mew=1.5)
pb.plot(data[:,2],data[:,5],'kx',mew=1.5)
pb.ylabel('falling time'), pb.xlabel(names[2])

pb.subplot(224)
pb.plot(data[:,3],data[:,4],'kx',mew=1.5)
pb.plot(data[:,3],data[:,5],'kx',mew=1.5)
pb.ylabel('falling time'), pb.xlabel(names[3])

## correlation between throws
pb.figure()
pb.plot(data[:,4],data[:,5],'kx',mew=1.5)
pb.ylabel('experiment 2'), pb.xlabel('experiment 1')
pb.title('flight duration')

varNoise = .5*np.var(data[:,4]-data[:,5])


## More imaginative : plot the helicopters (side view) vivid colors mean long flight duration

## coordinate change
def angle(X):
	# X is Wing-length, Wing-width, Tail-length, Arm-length
	# returns the angle (in degrees) between the tail and the wing
	return(180/np.pi*np.arccos(-1.*((X[:,3]-2.5)**2-(X[:,2]-2.5)**2-X[:,0]**2)/(2*(X[:,2]-2.5)*X[:,0])))

alpha = angle(X)*np.pi/180

pb.figure()
for i in range(30):    
    X2 = np.zeros((3,2))
    X2[1,:] = X[i,0]*np.asarray((np.cos(alpha[i]),np.sin(alpha[i])))
    X2[2,0] = X[i,2]-2.5
    pb.plot(X2[:,1],-X2[:,0],'g',mew=2,alpha=(F[i]-np.min(F))/(np.max(F)-np.min(F)))
    pb.plot(-X2[:,1],-X2[:,0],'g',mew=2,alpha=(F[i]-np.min(F))/(np.max(F)-np.min(F)))
    pb.plot([0,0],[0,-X[i,2]],'g',mew=2,alpha=(F[i]-np.min(F))/(np.max(F)-np.min(F)))
    
pb.axis([-7,7,-10,3])

#########################
## Question 2

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x[:,1:4]
	b2 = x[:,0:1]*x[:,3:4]
	#b3 = b1**2
	#b4 = (x[:,1])[:,None]
	B = np.hstack((b0,b1,b2))
	return(B)

def LR(X,F,B,tau2):
	#input:	  X, np.array with d columns representing the DoE
	#		  F, np.array with 1 column representing the observations
	#		  B, a function returning the (p) basis functions evaluated at x
	# 		  tau2, noise variance
	#output:  beta, estimate of coefficients np.array of shape (p,1)
	#		  covBeta, cov matrix of beta, np.array of shape (p,p)
	BX = B(X)
	covBeta = np.linalg.inv(np.dot(BX.T,BX))
	beta = np.dot(covBeta,np.dot(BX.T,F))
	return(beta,tau2*covBeta)

#########################
## Question 3

def predLR(x,B,beta,covBeta):
	#function returning predicted mean and variance
	#input:	  x, np.array with d columns representing m prediction points
	#		  B, a function returning the (p) basis functions evaluated at x
	#		  beta, estimate of the regression coefficients
	# 		  covBeta, covariance matrix of beta
	#output:  m, predicted mean at x, np.array of shape (m,1)
	#		  v, predicted variance, np.array of shape (m,1)
	m = np.dot(B(x),beta)
	v = np.dot(B(x),np.dot(covBeta,B(x).T))
	return(m,v)

def plotModel(x,m,v):
	#input:	  x, np.array with d columns representing m prediction points
	#		  m, predicted mean at x, np.array of shape (m,1)
	#		  v, predicted variance matrix, np.array of shape (m,m)
    x = x.flatten()
    m = m.flatten()
    v = np.diag(v)
    upper=m+2*np.sqrt(v)
    lower=m-2*np.sqrt(v)
    pb.plot(x,m,color="#204a87",linewidth=2)
    pb.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color="#729fcf",alpha=0.3)
    pb.plot(x,upper,color="#204a87",linewidth=0.2)
    pb.plot(x,lower,color="#204a87",linewidth=0.2)

def R2(X,F,B,beta):
	return(1-sum((F-np.dot(B(X),beta))**2)/sum((F-np.mean(F))**2))

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x[:,0:1]
	b2 = x[:,0:1]**2
	B = np.hstack((b0,b1,b2))
	return(B)

## predict
g = np.linspace(-0.2,1.2,100)[:,None]
x = np.hstack((g,g,g,g))
beta,covBeta = LR(XS,F,B,varNoise)
m,v = predLR(x,B,beta,covBeta)

## plot model
pb.figure()
plotModel(g,m,v)
pb.plot(XS[:,0],F,'kx',mew=1.5)

## compute R2
print "R2 = ", round(R2(XS,F,B,beta)[0],2)

#########################
## Question 4

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x
	b2 = x**2
	B = np.hstack((b0,b1,b2))
	return(B)

beta,covBeta = LR(X,F,B,varNoise)

def pvalue(beta,covBeta,X):
	df = X.shape[0] - len(beta)
	cdf = stats.t.cdf(np.abs(beta[:,0])/np.sqrt(np.diag(covBeta)),df)
	return(2*(1 - cdf))

## compute R2
print "R2 = ", round(R2(X,F,B,beta)[0],2)
print "p-values = ", np.round(pvalue(beta,covBeta,X),2)



#########################
## Question 5

## mapping to the new space
def old2new(X):
	Y = 0*X
	Y[:,0] = angle(X)
	Y[:,1] = X[:,0] * X[:,1]  
	Y[:,2] = X[:,0] + X[:,2] + X[:,3] 
	Y[:,3] = X[:,0] / (X[:,2] - 2.5)
	return(Y)

## mapping to the original space
def new2old(Y):
	X = 0*Y
	f_1 = np.sqrt(1+(1/Y[:,3])**2-2*np.cos(Y[:,0]*np.pi/180)*(1/Y[:,3]))
	X[:,0] = (Y[:,2]-5) / (1 + 1./Y[:,3] + f_1)
	X[:,1] = Y[:,1] / X[:,0]
	X[:,2] = X[:,0] / Y[:,3] + 2.5
	X[:,3] = Y[:,2] - X[:,0] - X[:,2]
	return(X)

Xnew = old2new(X)
namesNew = ['wing angle', 'wing area','total length', 'wing_l / tail_l ratio']

## outputs versus inputs
pb.figure(figsize=(10,10))

pb.subplot(221)
pb.plot(Xnew[:,0:1],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[0]), pb.ylabel('falling time')

pb.subplot(222)
pb.plot(Xnew[:,1:2],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[1]), pb.ylabel('falling time')

pb.subplot(223)
pb.plot(Xnew[:,2:3],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[2]), pb.ylabel('falling time')

pb.subplot(224)
pb.plot(Xnew[:,3:4],data[:,4:6],'kx',mew=1.5)
pb.xlabel(namesNew[3]), pb.ylabel('falling time')


def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x
	b2 = x[:,3:4]**2
	B = np.hstack((b0,b1,b2))
	return(B)

beta,covBeta = LR(Xnew,F,B,varNoise)

## model validation
print "R2 = ", round(R2(Xnew,F,B,beta)[0],2)
print "p-values = ", np.round(pvalue(beta,covBeta,Xnew),2)


m,v = predLR(Xnew,B,beta,covBeta)

pb.figure()
pb.plot(m,F,'kx')

## residuals versus inputs
pb.figure(figsize=(10,10))

pb.subplot(221)
pb.plot(X[:,0:1],m-F,'kx',mew=1.5)
pb.ylabel('residuals'), pb.xlabel(namesNew[0])

pb.subplot(222)
pb.plot(X[:,1:2],m-F,'kx',mew=1.5)
pb.ylabel('residuals'), pb.xlabel(namesNew[1])

pb.subplot(223)
pb.plot(X[:,2:3],m-F,'kx',mew=1.5)
pb.ylabel('residuals'), pb.xlabel(namesNew[2])

pb.subplot(224)
pb.plot(X[:,3:4],m-F,'kx',mew=1.5)
pb.ylabel('residuals'), pb.xlabel(namesNew[3])


#########################
## Question 6

## domain of interest
limits = np.array([75,115,20,35,22,32,0.65,1.6]).reshape(4,2).T

## hypercube
pb.figure(figsize=(10,10))

pb.subplot(221)
pb.plot(Xnew[:,0:1],data[:,4:6],'kx',mew=1.5)
pb.axvspan(limits[0,0], limits[1,0], color='b', alpha=0.3, lw=0)
pb.xlabel(namesNew[0]), pb.ylabel('falling time')

pb.subplot(222)
pb.plot(Xnew[:,1:2],data[:,4:6],'kx',mew=1.5)
pb.axvspan(limits[0,1], limits[1,1], color='b', alpha=0.3, lw=0)
pb.xlabel(namesNew[1]), pb.ylabel('falling time')

pb.subplot(223)
pb.plot(Xnew[:,2:3],data[:,4:6],'kx',mew=1.5)
pb.axvspan(limits[0,2], limits[1,2], color='b', alpha=0.3, lw=0)
pb.xlabel(namesNew[2]), pb.ylabel('falling time')

pb.subplot(224)
pb.plot(Xnew[:,3:4],data[:,4:6],'kx',mew=1.5)
pb.axvspan(limits[0,3], limits[1,3], color='b', alpha=0.3, lw=0)
pb.xlabel(namesNew[3]), pb.ylabel('falling time')


