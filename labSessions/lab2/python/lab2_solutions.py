import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import sobol_seq

plt.ion()

##################################################################
##                  part 1: useful functions                    ##
##################################################################

####################################
## Examples of DoE

# random uniform DoE
X = np.random.uniform(0,1,(40,2))

plt.figure(figsize=(4,4))
plt.plot(X[:,0],X[:,1],'kx')
plt.title('random uniform')

## Sobol DoE
XS = sobol_seq.i4_sobol_generate(2,40)

plt.figure(figsize=(4,4))
plt.plot(XS[:,0],XS[:,1],'kx')
plt.title('Sobol sequence')

####################################
# Space filling criteria
def discrepancy(X):
	# compute the discrepancy with respect to the center of the domain
	n,d = X.shape
	Xcentred = X-.5
	distCentreX = np.sort(np.max(np.abs(Xcentred),axis=1))
	theoreticalProba = (2*distCentreX)**d
	empiricalProba = 1.*np.arange(n)/n
	D = np.abs(np.hstack((theoreticalProba-empiricalProba,theoreticalProba-empiricalProba+1./n)))
	return(np.max(D))

discrepancy(X)
discrepancy(XS)

def maximin(X):
	n,d = X.shape
	distMat = np.sqrt(np.sum((X[:,None,:] - X[None,:,:])**2,axis=2))
	distMat += np.sqrt(d)*np.eye(n)
	return(np.min(distMat))

maximin(X)
maximin(XS)

def minimax(X):
	n,d = X.shape
	G = sobol_seq.i4_sobol_generate(d,10000)
	dXG = np.sum((X[:,None,:]-G[None,:,:])**2,axis=2)
	dXGmin = np.min(dXG,axis=0)
	minimax2 = np.max(dXGmin)
	return(np.sqrt(minimax2))

minimax(X)
minimax(XS)

def IMSE(X,theta=.2):
	# squared exponential kernel is assumed
	n,d = X.shape
	G = sobol_seq.i4_sobol_generate(d, 10000)
	dX2 = np.sum((X[:,None,:]-X[None,:,:])**2/theta**2,2)
	dG2 = np.sum((G[:,None,:]-X[None,:,:])**2/theta**2,2)
	kX_1 = np.linalg.inv(np.exp(-dX2/2.))
	kG = np.exp(-dG2/2.)
	imse = 1 - np.mean(np.sum(np.dot(kG,kX_1)*kG,axis=1))
	return(imse)

IMSE(X)
IMSE(XS)

##################################################################
##                      part 2: lab session                     ##
##################################################################

#################
## Q1

def LHS(n,d):
	x = np.linspace(.5/n,1-.5/n,n)
	X = np.zeros((n,d))
	for i in range(d):
		X[:,i] = np.random.permutation(x)
	return(X)

X = LHS(20,2)

plt.figure()
plt.plot(X[:,0],X[:,1],'kx',mew=1.5)
plt.title('LHS')

#################
## Q2
def CVT(n,d):
	x = np.random.uniform(0,1,(1000,d))
	X = KMeans(n_clusters=n).fit(x).cluster_centers_
	return(X)

X = CVT(20,2)

plt.figure()
plt.plot(X[:,0],X[:,1],'kx',mew=1.5)
plt.title('CVT')

#################
## Q3

Xunf = np.random.uniform(0,1,(30,4))
Xlhs = LHS(30,4)
Xcvt = CVT(30,4)
Xsob = sobol_seq.i4_sobol_generate(4,30)

DOE = [Xunf, Xlhs, Xcvt, Xsob]
CRIT = [discrepancy, maximin, minimax, IMSE]

n_doe = len(DOE)
n_crit = len(CRIT)
RES = np.zeros((n_doe,n_crit))
for i in range(n_doe):
	for j in range(n_crit):
		RES[i,j] = CRIT[j](DOE[i])

np.round(RES,2)

###############
##
X = Xcvt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])
