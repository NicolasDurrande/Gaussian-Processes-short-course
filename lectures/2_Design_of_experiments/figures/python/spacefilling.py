import numpy as np
import pylab as pb
import random
pb.ion()

params = {'backend': 'ps',
		  'axes.labelsize': 20,
          'text.fontsize': 20,
          'legend.fontsize': 20,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True}
pb.rcParams.update(params)

colorsHex = {\
"Aluminium6":"#2e3436",\
"Aluminium5":"#555753",\
"Aluminium4":"#888a85",\
"Aluminium3":"#babdb6",\
"Aluminium2":"#d3d7cf",\
"Aluminium1":"#eeeeec",\
"lightPurple":"#ad7fa8",\
"mediumPurple":"#75507b",\
"darkPurple":"#5c3566",\
"lightBlue":"#729fcf",\
"mediumBlue":"#3465a4",\
"darkBlue": "#204a87",\
"lightGreen":"#8ae234",\
"mediumGreen":"#73d216",\
"darkGreen":"#4e9a06",\
"lightChocolate":"#e9b96e",\
"mediumChocolate":"#c17d11",\
"darkChocolate":"#8f5902",\
"lightRed":"#ef2929",\
"mediumRed":"#cc0000",\
"darkRed":"#a40000",\
"lightOrange":"#fcaf3e",\
"mediumOrange":"#f57900",\
"darkOrange":"#ce5c00",\
"lightButter":"#fce94f",\
"mediumButter":"#edd400",\
"darkButter":"#c4a000"}

darkList = [colorsHex['darkBlue'],colorsHex['darkRed'],colorsHex['darkGreen'], colorsHex['darkOrange'], colorsHex['darkButter'], colorsHex['darkPurple'], colorsHex['darkChocolate'], colorsHex['Aluminium6']]
mediumList = [colorsHex['mediumBlue'], colorsHex['mediumRed'],colorsHex['mediumGreen'], colorsHex['mediumOrange'], colorsHex['mediumButter'], colorsHex['mediumPurple'], colorsHex['mediumChocolate'], colorsHex['Aluminium5']]
lightList = [colorsHex['lightBlue'], colorsHex['lightRed'],colorsHex['lightGreen'], colorsHex['lightOrange'], colorsHex['lightButter'], colorsHex['lightPurple'], colorsHex['lightChocolate'], colorsHex['Aluminium4']]

tango = darkList + mediumList + lightList
############################################
# Volume of 0.9 hypercube
d = np.linspace(1,100)
V = .9**d

pb.figure(figsize=(5,3))
pb.plot(d,V,'bx',mew=2)
pb.ylim((0,1.1))
pb.xlabel('dimension $d$')
pb.ylabel('Volume in blue area')

pb.savefig('spf_volume.pdf',bbox_inches='tight')

############################################
# Volume of 0.9 hypercube
n = 50
X = np.random.uniform(0,1,(10,2))
g = np.linspace(0,1,n)
G = np.zeros((n**2,2))
for i in range(n):
	for j in range(n):
		G[50*i+j,:] = (g[i],g[j])


# maximin
dX = np.sum((X[:,None,:]-X[None,:,:])**2,axis=2)
dX += 2*np.eye(10)
np.min(dX)
imin = np.argmin(dX)/10
jmin = np.argmin(dX) - imin*10

# minimax
dXG = np.sum((X[:,None,:]-G[None,:,:])**2,axis=2)
dXGmin = np.min(dXG,axis=0)
jmin2 = np.argmax(dXGmin)
imin2 = np.argmin(dXG[:,jmin2])


pb.figure(figsize=(5,5))
pb.plot(X[:,0],X[:,1],'kx',mew=3)
pb.plot((X[imin,0],X[jmin,0]),(X[imin,1],X[jmin,1]),linewidth=2)
pb.plot((X[imin2,0],G[jmin2,0]),(X[imin2,1],G[jmin2,1]),linewidth=2)
pb.xlim((0,1))
pb.ylim((0,1))
pb.legend(('design','maximin','minimax'))
pb.savefig('spf_minimaxmaximin.pdf',bbox_inches='tight')


## Halton
X1 = (1/2., 1/4., 3/4., 1/8., 5/8., 3/8., 7/8., 1/16., 9/16.)
X2 = (1/3., 2/3., 1/9., 4/9., 7/9., 2/9., 5/9., 8/9., 1/27.)

pb.figure(figsize=(5,5))
pb.plot(X1,X2,'kx',mew=3)
pb.xlim((0,1))
pb.ylim((0,1))
pb.savefig('spf_halton.pdf',bbox_inches='tight')

##########################################
##  CVT

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


U = np.random.uniform(0,1,(1000,2))
(Ut,ind) = find_centers(U, 10)
Ut = np.asarray(Ut)

pb.figure(figsize=(5,5))
pb.plot(U[:,0],U[:,1],'kx',mew=2)
pb.xlim((0,1))
pb.ylim((0,1))
pb.savefig('spf_kmeans1.pdf',bbox_inches='tight')


pb.figure(figsize=(5,5))
for i in range(10):
	indi = np.asarray(ind[i])
	pb.plot(indi[:,0],indi[:,1],'x',mew=2,color=tango[i])
	pb.plot(Ut[i,0],Ut[i,1],'*',mew=1,color=tango[i],markersize=20)

pb.xlim((0,1))
pb.ylim((0,1))
pb.savefig('spf_kmeans2.pdf',bbox_inches='tight')


def McQueen(k):
    nit=1000
    X = np.random.uniform(0,1,(k,2))
    P = [X[i,:] for i in range(k)]
    for i in range(nit):
        z = np.random.uniform(0,1,(1,2))
        j = np.argmin(np.sum((X[:,None,:]-z[None,:,:])**2,axis=2))
        P[j] = np.vstack((P[j],z))
        X[j,:]=(P[j].shape[0]*X[j,:]+z)/(P[j].shape[0]+1)
    return(X, P)

(Ut,ind) = McQueen(10)

pb.figure(figsize=(5,5))
for i in range(10):
	indi = ind[i]
	pb.plot(indi[:,0],indi[:,1],'x',mew=2,color=tango[i])
	pb.plot(Ut[i,0],Ut[i,1],'*',mew=1,color=tango[i],markersize=20)

pb.xlim((0,1))
pb.ylim((0,1))
pb.savefig('spf_McQueen.pdf',bbox_inches='tight')

