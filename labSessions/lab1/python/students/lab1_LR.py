import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
plt.ion()

#########################
## load the data
# first 4 columns are inputs
# last column is output
data = np.genfromtxt('lab1_data.csv',delimiter=',')

X = data[:,0:4]
F = data[:,4:5]

names = ["x" + str(i) for i in range(4)]

#########################
## Question 1


#########################
## Question 2

def B(x):
	# function returning the matrix of basis functions evaluated at x
	#input:	  x, np.array with d columns
	#output:  a matrix of geberal term B_{i,j} = b_j(x_i)
	b0 = np.ones((x.shape[0],1))
	b1 = (x[:,0])[:,None]
	B = np.hstack((b0,b1))
	return(B)

def LR(X,F,B):
	#input:	  X, np.array with d columns representing the DoE
	#		  F, np.array with 1 column representing the observations
	#		  B, a function returning the (p) basis functions evaluated at x
	#output:  beta, estimate of coefficients np.array of shape (p,1)
	#		  covBeta, cov matrix of beta, np.array of shape (p,p)

	# ... to be completed ...

	return(beta,covBeta)

#########################
## Question 3

def predLR(x,B,beta,covBeta):
	#function returning predicted mean and variance
	#input:	  x, np.array with d columns representing m prediction points
	#		  B, a function returning the (p) basis functions evaluated at x
	#		  beta, estimate of the regression coefficients
	# 		  covBeta, covariance matrix of beta
	#output:  m, predicted mean at x, np.array of shape (m,1)
	#		  v, predicted variance matrix, np.array of shape (m,m)

	# ... to be completed ...

	return(m,v)

def R2(X,F,B,beta):
	return(1-sum((F-np.dot(B(X),beta))**2)/sum((F-np.mean(F))**2))

def pvalue(beta,covBeta,X):
	df = X.shape[0] - len(beta)
	cdf = stats.t.cdf(np.abs(beta[:,0])/np.sqrt(np.diag(covBeta)),df)
	return(2*(1 - cdf))

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

#########################
## Question 4

#########################
## Question 5

#########################
## Question 6
