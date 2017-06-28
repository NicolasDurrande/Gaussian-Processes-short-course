from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats
plt.ion()

#########################
## load the data
# first 4 columns are inputs
# last column is output
data = np.genfromtxt('lab1_data.csv',delimiter=',')

X = data[:,0:4]
F = data[:,4:5]

names = ["x" + str(i) for i in range(1,5)]

#########################
## Question 1

## outputs versus inputs
plt.figure(figsize=(10,10))
for i in range(4):
	plt.subplot(2,2,i+1)
	plt.plot(data[:,i],data[:,4],'kx',mew=1.5)
	plt.ylabel('distance')
	plt.xlabel(names[i])

## plot interactions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], F)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,2], F)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1], X[:,3], F)

#########################
## Question 2

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x[:,1:4]
	B = np.hstack((b0,b1))
	return(B)

def LR(X,F,B):
	#input:	  X, np.array with d columns representing the DoE
	#		  F, np.array with 1 column representing the observations
	#		  B, a function returning the (p) basis functions evaluated at x
	#output:  beta, estimate of coefficients np.array of shape (p,1)
	#		  covBeta, cov matrix of beta, np.array of shape (p,p)
	BX = B(X)
	n,p = BX.shape
	covBeta = np.linalg.inv(np.dot(BX.T,BX))
	beta = np.dot(covBeta,np.dot(BX.T,F))
	tau2 = sum((np.dot(B(X),beta)-F)**2)/(n-p)
	return(beta,tau2*covBeta)

beta, covbeta = LR(X,F,B)

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
    plt.plot(x,m,color="#204a87",linewidth=2)
    plt.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color="#729fcf",alpha=0.3)
    plt.plot(x,upper,color="#204a87",linewidth=0.2)
    plt.plot(x,lower,color="#204a87",linewidth=0.2)

def R2(X,F,B,beta):
	r2 = 1-sum((F-np.dot(B(X),beta))**2)/sum((F-np.mean(F))**2)
	return(r2[0])

def pvalue(beta,covBeta,X):
	df = X.shape[0] - len(beta)
	cdf = stats.t.cdf(np.abs(beta[:,0])/np.sqrt(np.diag(covBeta)),df)
	return(2*(1 - cdf))

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x[:,0:1]
	b2 = b1**2
	B = np.hstack((b0,b1,b2))
	return(B)

## predict
x = np.linspace(-0.2,1.2,100)[:,None]
beta,covBeta = LR(X[:,1:2],F,B)
m,v = predLR(x,B,beta,covBeta)

plt.figure()
plotModel(x,m,v)
plt.plot(X[:,1],F,'kx',mew=1.5)

## compute R2
print "R2 = ", round(R2(X[:,1:2],F,B,beta),2)
print "p-values = ", np.round(pvalue(beta,covBeta,X),3)

## plot all models
plt.figure(figsize=(10,10))
for i in range(4):
	ax = plt.subplot(2,2,i+1)
	beta,covBeta = LR(X[:,i:i+1],F,B)
	m,v = predLR(x,B,beta,covBeta)
	r2 = np.round(R2(X[:,i:i+1],F,B,beta),2)
	plotModel(x,m,v)
	plt.plot(X[:,i],F,'kx',mew=1.5)
	plt.text(0.1, 0.9,'R2 = {}'.format(r2), ha='left', va='center', transform=ax.transAxes)
	plt.xlabel(names[i])

#########################
## Question 4

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x[:,0:2]
	b2 = b1**2
	b3 = b1**3
	B = np.hstack((b0,b1,b2,b3))
	return(B)

beta,covBeta = LR(X,F,B)

## compute R2
print "R2 = ", round(R2(X,F,B,beta),2)
print "p-values = ", np.round(pvalue(beta,covBeta,X),3)

g = np.linspace(0,1,21)
G = np.meshgrid(g,g)
gg = np.hstack((G[0].flatten()[:,None],G[1].flatten()[:,None]))

m,v = predLR(gg,B,beta,covBeta)
M = m.reshape((len(g),len(g)))
V = np.diag(v).reshape((len(g),len(g)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], F)
ax.plot_surface(G[0],G[1],M,rstride=1,cstride=1,alpha=.5,color="red")

# add confidence intervals
ax.plot_surface(G[0], G[1], M+2*np.sqrt(V),
                rstride=1, cstride=1,
                linewidth=0, alpha=.2, color="blue")
ax.plot_surface(G[0], G[1], M-2*np.sqrt(V),
                rstride=1, cstride=1,
				linewidth=0, alpha=.2, color="blue")

#########################
## Question 5

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix (b_j(x_i))_{i,j}
	b0 = np.ones((x.shape[0],1))
	b1 = x[:,0:1] - x[:,1:2]
	b2 = b1**2
	b3 = b1**3
	B = np.hstack((b0,b1,b2,b3))
	return(B)

beta, covBeta = LR(X,F,B)

print "R2 = ", round(R2(X,F,B,beta),2)
print "p-values = ", np.round(pvalue(beta,covBeta,X),3)

#######################
## predict (1d)
g = np.linspace(0,1,100)[:,None]
x = np.hstack((g,1-g))
beta,covBeta = LR(X,F,B)
m,v = predLR(x,B,beta,covBeta)

## plot 1d model
plt.figure()
plotModel(x[:,0]-x[:,1],m,v)
plt.plot(X[:,0]-X[:,1],F,'kx',mew=1.5)


#######################
## predict (2d)
g = np.linspace(0,1,21)
G = np.meshgrid(g,g)
gg = np.hstack((G[0].flatten()[:,None],G[1].flatten()[:,None]))

m,v = predLR(gg,B,beta,covBeta)
M = m.reshape((len(g),len(g)))
V = np.diag(v).reshape((len(g),len(g)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], F)
ax.plot_surface(G[0],G[1],M,rstride=1,cstride=1,alpha=.5,color="red")

# add confidence intervals
ax.plot_surface(G[0], G[1], M+2*np.sqrt(V),
                rstride=1, cstride=1,
                linewidth=0, alpha=.2, color="blue")
ax.plot_surface(G[0], G[1], M-2*np.sqrt(V),
                rstride=1, cstride=1,
				linewidth=0, alpha=.2, color="blue")


#########################
## Question 6
m,v = predLR(X,B,beta,covBeta)

## residuals versus new 1D input and output
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.plot(X[:,0]-X[:,1],F-m,'kx',mew=1.5)
plt.xlabel("input")
plt.ylabel('residuals')

plt.subplot(122)
plt.plot(X[:,0]-X[:,1],F-m,'kx',mew=1.5)
plt.xlabel("output")
plt.ylabel('residuals')

## residuals versus original inputs
plt.figure(figsize=(10,10))

plt.subplot(221)
plt.plot(X[:,0],F-m,'kx',mew=1.5)
plt.xlabel(names[0])
plt.ylabel('residuals')

plt.subplot(222)
plt.plot(X[:,1],F-m,'kx',mew=1.5)
plt.xlabel(names[1])
plt.ylabel('residuals')

plt.subplot(223)
plt.plot(X[:,2],F-m,'kx',mew=1.5)
plt.xlabel(names[2])
plt.ylabel('residuals')

plt.subplot(224)
plt.plot(X[:,3],F-m,'kx',mew=1.5)
plt.xlabel(names[3])
plt.ylabel('residuals')
