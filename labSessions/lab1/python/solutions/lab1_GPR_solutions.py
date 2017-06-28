import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#########################
## Question 1

# the input points X and Y are always arrays with d columns
def kernGauss(X,Y,sigma2=1.,theta=.2):
	d2 = np.sum((X[:,None,:]-Y[None,:,:])**2/theta**2,2)
	k = sigma2*np.exp(-d2/2.)
	return(k)

def kernMat32(X,Y,sigma2=1.,theta=.2):
	d = np.sqrt(np.sum((X[:,None,:]-Y[None,:,:])**2/theta**2,2))
	k = sigma2*(1+np.sqrt(3)*d)*np.exp(-np.sqrt(3)*d)
	return(k)

def kernMat52(X,Y,sigma2=1.,theta=.2):
	d = np.sqrt(np.sum((X[:,None,:]-Y[None,:,:])**2/theta**2,2))
	k = sigma2*(1+np.sqrt(5)*d+5./3.*d**2)*np.exp(-np.sqrt(5)*d)
	return(k)

def kernCst(X,Y,sigma2=1.):
	k = sigma2*np.ones((X.shape[0],Y.shape[0]))
	return(k)

def kernBrown(X,Y,sigma2=1.):
	k = sigma2*np.fmin(X,Y.T)
	return(k)

def kernWhiteNoise(X,Y,sigma2=1.):
	k = sigma2*np.all(X[:,None,:]==Y[None,:,:],axis=2)
	return(k)

kern = kernMat32

## plot kernel
x = np.linspace(-1,1,200)[:,None]
y = kern(x,np.zeros((1,1)),1,.2)
plt.plot(x,y,linewidth=2)

#########################
## Question 2
def sampleGP(x,mu,kern,n,**kwargs):
	# return n sample paths from a GP N(mu(.),kern(.,.)) evaluated at x
	m = mu(x)
	K = kern(x,x,**kwargs)
	Kv, KV = np.linalg.eig(K)
	N = np.random.normal(0,1,(x.shape[0],n))
	Z = np.dot(np.dot(KV,np.diag(np.sqrt(Kv))),N)
	return(m+Z)

def mu(x):
	return(0*x)

#########################
## Question 3

x = np.linspace(0,1,200)[:,None]

Z = sampleGP(x,mu,kern,3,sigma2=1.,theta=.2)

plt.plot(x,Z)

#########################
## Question 4
def GPR(x,X,F,kern,**kwargs):
	# return the mean predictor (m(x)=E[Z(x)|Z(X)=F]) and the conditional covariance matrix cov[Z(x),Z(x)|Z(X)=F]
	# We assume here that Z is centred
	Kxx = kern(x,x,**kwargs)
	KxX = kern(x,X,**kwargs)
	KXX_1 = np.linalg.inv(kern(X,X,**kwargs))
	mean = np.dot(np.dot(KxX,KXX_1),F)
	var = Kxx - np.dot(np.dot(KxX,KXX_1),KxX.T)
	return(mean,var)

#########################
## Question 5
def ftest(x):
	return(np.sin(3*np.pi*x)+2*x)

def plotModel(x,m,v,**kwargs):
    x = x.flatten()
    m = m.flatten()
    v = np.diag(v)
    upper=m+2*np.sqrt(v)
    lower=m-2*np.sqrt(v)
    plt.plot(x,m,color="#204a87",linewidth=2,**kwargs)
    plt.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color="#729fcf",alpha=0.3)
    plt.plot(x,upper,color="#204a87",linewidth=0.2)
    plt.plot(x,lower,color="#204a87",linewidth=0.2)

x = np.linspace(0,1,200)[:,None]
X = np.linspace(.1,.9,5)[:,None]
F = ftest(X)

condMean, condVar = GPR(x,X,F,kern,sigma2=1,theta=.2)

plotModel(x,condMean,condVar)
plt.plot(x,ftest(x))
