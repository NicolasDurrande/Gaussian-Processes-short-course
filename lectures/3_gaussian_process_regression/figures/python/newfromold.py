import numpy as np
import pylab as pb
import GPy

pb.ion()
pb.close('all')
###############################################
x1 = np.linspace(0,1,100)
x2 = np.linspace(0,1,100)

f1 = np.sin(2*np.pi*x1)
f2 = 2*x2

#### univariate
pb.figure(figsize=(5,5))
pb.plot(x1,f1)
pb.ylim((-1.2,1.2))
pb.savefig('newfromold-f1.pdf',bbox_inches='tight')

pb.figure(figsize=(5,5))
pb.plot(x1,f2)
pb.ylim((-.2,2.2))
pb.savefig('newfromold-f2.pdf',bbox_inches='tight')

#### Sum 1
pb.figure(figsize=(5,5))
pb.plot(x1,f1+ f2)
pb.ylim((-.2,2.2))
pb.savefig('newfromold-sum1.pdf',bbox_inches='tight')

#### Sum 2
xg1 = np.linspace(0,1,20)
xg2 = np.linspace(0,1,20)

X1, X2 = np.meshgrid(xg1, xg2)
F = np.sin(2*np.pi*X1) + 2*X2

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X1,X2, F, rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-sum2.pdf',bbox_inches='tight')

###############################################
###############################################
## Sum of kernels same space

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=1.5)
kg2 = GPy.kern.exponential(input_dim=1,variance=.1,lengthscale=.05)

k = kg1 + kg2

## Gauss exp
pb.figure(figsize=(3,3))
ax = pb.subplot(111)
kg1.plot(resolution=401,color='b')
ax.set_xticklabels([])
ax.set_yticklabels([])
#pb.title('squared exponential')
pb.xlabel('')
pb.ylabel('')
pb.ylim((0,1.25))
pb.savefig('newfromold-a.pdf',bbox_inches='tight')

pb.figure(figsize=(3,3))
ax = pb.subplot(111)
kg2.plot(resolution=401,color='b')
ax.set_xticklabels([])
ax.set_yticklabels([])
#pb.title('exponential')
pb.xlabel('')
pb.ylabel('')
pb.ylim((0,1.25))
pb.savefig('newfromold-b.pdf',bbox_inches='tight')

pb.figure(figsize=(3,3))
ax = pb.subplot(111)
k.plot(resolution=401)
ax.set_xticklabels([])
ax.set_yticklabels([])
#pb.title('psd kernel')
pb.xlabel('')
pb.ylabel('')
pb.ylim((0,1.25))
pb.savefig('newfromold-sumab1.pdf',bbox_inches='tight')

########################
x = np.linspace(0,3,1000)[:,None]
K = k.K(x)

Z = np.random.multivariate_normal(0*x[:,0],K,10)

pb.figure(figsize=(9,3))
pb.plot(x,Z[0:3,:].T)
pb.savefig('newfromold-sumabtraj.pdf',bbox_inches='tight')

###############################################
###############################################
## Sum of kernels different space

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)
kg2 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)

xg = np.linspace(0,1,101)

##
km1 = kg1.K(xg[:,None],np.zeros((1,1))+0.5)

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot(xg,xg*0, km1[:,0])
pb.ylim((0,1))
pb.savefig('newfromold-sum2-k1.pdf',bbox_inches='tight')

##
km2 = kg2.K(xg[:,None],np.zeros((1,1))+0.5)

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot(xg*0,xg, km2[:,0])
pb.xlim((0,1))
pb.savefig('newfromold-sum2-k2.pdf',bbox_inches='tight')

##
xg = np.linspace(0,1,30)
X,Y = np.meshgrid(xg,xg)

km1 = kg1.K(xg[:,None],np.zeros((1,1))+0.5)
km2 = kg2.K(xg[:,None],np.zeros((1,1))+0.5)

Z = km1 + km2.T

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-sum2-k12.pdf',bbox_inches='tight')

##################
## Simulate sample paths
X,Y = np.meshgrid(xg,xg)
XX = np.hstack((X.flatten()[:,None],Y.flatten()[:,None]))

k = kg1.add(kg2,tensor=1)
K = k.K(XX)

Z = np.random.multivariate_normal(0*XX[:,0],K,10)
Z = Z.reshape((10,30,30))

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z[4,:,:], rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-sum2-traj124.pdf',bbox_inches='tight')

####################################################
####################################################
## additive vs product test function
def ftest(X):
	return(np.sin(4*np.pi*X[:,0]) + np.cos(4*np.pi*X[:,1]) + 2*X[:,1] )

xg = np.linspace(0,1,31)
XX,YY = np.meshgrid(xg,xg)

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)
kg2 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)

X = np.random.uniform(0,1,(20,2))
Y = ftest(X)[:,None]

Xg,Yg = np.meshgrid(xg,xg)
XX = np.hstack((Xg.flatten()[:,None],Yg.flatten()[:,None]))

Yt = ftest(XX)[:,None]
# plot
fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(Xg,Yg,Yt.reshape((31,31)), rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-productvssum2-predt.pdf',bbox_inches='tight')

# plot DOE
fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot(X[:,0],X[:,1],Y[:,0], 'kx')
pb.savefig('newfromold-productvssum2-preddoe.pdf',bbox_inches='tight')

#####################
# prod
# kernel
kp = kg1.prod(kg2,tensor=True)
# model
mp = GPy.models.GPRegression(X,Y,kp)
mp.ensure_default_constraints()
mp.constrain_fixed('noise',1e-3)
mp.optimize()
# predict 
Yp,Vp,_,_ = mp.predict(XX)
RMSEp = np.sqrt(np.mean((Yp-Yt)**2))
# plot
fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(Xg,Yg,Yp.reshape((31,31)), rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-productvssum2-predp.pdf',bbox_inches='tight')


#####################
# add
# kernel
ka = kg1.add(kg2,tensor=True)
# model
ma = GPy.models.GPRegression(X,Y,ka)
ma.ensure_default_constraints()
ma.constrain_fixed('noise',1e-3)
ma.optimize()
# predict 
Ya,Va,_,_ = ma.predict(XX)
RMSEa = np.sqrt(np.mean((Ya-Yt)**2))
# plot
fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(Xg,Yg,Ya.reshape((31,31)), rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-productvssum2-preda.pdf',bbox_inches='tight')


####################################################
####################################################
## additive vs product
xg = np.linspace(0,1,51)
X,Y = np.meshgrid(xg,xg)

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.05)
kg2 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.05)

km1 = kg1.K(xg[:,None],np.zeros((1,1))+0.5)
km2 = kg2.K(xg[:,None],np.zeros((1,1))+0.5)

Z = km1 * km2.T

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-product2-k12.pdf',bbox_inches='tight')

## additive vs product
xg = np.linspace(0,1,51)
X,Y = np.meshgrid(xg,xg)

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.05)
kg2 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.05)

km1 = kg1.K(xg[:,None],np.zeros((1,1))+0.5)
km2 = kg2.K(xg[:,None],np.zeros((1,1))+0.5)

Z = km1 + km2.T

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-productvssum2-k12.pdf',bbox_inches='tight')












##############################################################################
##############################################################################
## Product of kernels
##############################################################################
##############################################################################

###############################################
###############################################
## Product of kernels same space

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.33)
kg2 = GPy.kern.rbfcos(input_dim=1,variance=1.,frequencies=3.,bandwidths=.33)

xg = np.linspace(-1,1,101)[:,None]

## Gauss cos
pb.figure(figsize=(3,3))
ax = pb.subplot(111)
pb.plot(xg,kg1.K(xg,np.zeros((1,1))))
ax.set_xticklabels([])
ax.set_yticklabels([0])
pb.xlabel('')
pb.ylabel('')
pb.ylim((-.1,1.1))
pb.savefig('newfromold-pa.pdf',bbox_inches='tight')

pb.figure(figsize=(3,3))
ax = pb.subplot(111)
pb.plot(xg,np.cos(6*np.pi*xg))
ax.set_xticklabels([])
ax.set_yticklabels([])
pb.xlabel('')
pb.ylabel('')
pb.ylim((-1.2,1.2))
pb.savefig('newfromold-pb.pdf',bbox_inches='tight')

pb.figure(figsize=(3,3))
ax = pb.subplot(111)
pb.plot(xg,kg2.K(xg,np.zeros((1,1))))
ax.set_xticklabels([])
ax.set_yticklabels([])
pb.xlabel('')
pb.ylabel('')
pb.ylim((-1.2,1.2))
pb.savefig('newfromold-pab1.pdf',bbox_inches='tight')

########################
x = np.linspace(0,2,201)[:,None]
K = kg2.K(x)

Z = np.random.multivariate_normal(0*x[:,0],K,10)

pb.figure(figsize=(6,3))
pb.plot(x,Z[0:3,:].T)
pb.savefig('newfromold-pabtraj.pdf',bbox_inches='tight')

###############################################
###############################################
## PROD kernels different space

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)
kg2 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)

# xg = np.linspace(0,1,101)

# ##
# km1 = kg1.K(xg[:,None],np.zeros((1,1))+0.5)

# fig = pb.figure(figsize=(5,5))
# ax = fig.gca(projection='3d')
# ax.plot(xg,xg*0, km1[:,0])
# pb.ylim((0,1))
# pb.savefig('newfromold-sum2-k1.pdf',bbox_inches='tight')

# ##
# km2 = kg2.K(xg[:,None],np.zeros((1,1))+0.5)

# fig = pb.figure(figsize=(5,5))
# ax = fig.gca(projection='3d')
# ax.plot(xg*0,xg, km2[:,0])
# pb.xlim((0,1))
# pb.savefig('newfromold-sum2-k2.pdf',bbox_inches='tight')

##
xg = np.linspace(0,1,31)
X,Y = np.meshgrid(xg,xg)

km1 = kg1.K(xg[:,None],np.zeros((1,1))+0.5)
km2 = kg2.K(xg[:,None],np.zeros((1,1))+0.5)

Z = km1 * km2.T

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-prod2-k12.pdf',bbox_inches='tight')

##################
## Simulate sample paths
xg = np.linspace(0,1,30)
X,Y = np.meshgrid(xg,xg)
XX = np.hstack((X.flatten()[:,None],Y.flatten()[:,None]))

k = kg1.prod(kg2,tensor=1)
K = k.K(XX)

Z = np.random.multivariate_normal(0*XX[:,0],K,10)
Z = Z.reshape((10,30,30))

fig = pb.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z[1,:,:], rstride=1, cstride=1,cmap=pb.cm.coolwarm, linewidth=0.001, antialiased=False)
pb.savefig('newfromold-prod2-traj124.pdf',bbox_inches='tight')


##############################################################################
##############################################################################
## Multiplication by a function

km = GPy.kern.Matern32(1,1.,0.1)
########################
x = np.linspace(0.0001,1,200)[:,None]

fig = pb.figure(figsize=(5,5))

y = 0.15*np.ones((1,1))
pb.plot(x,km.K(x,y)/x/y)
pb.plot([y[0,0],y[0,0]],[0,(km.K(y,y)/y/y)[0,0]],'k:')

y = 0.25*np.ones((1,1))
pb.plot(x,km.K(x,y)/x/y)
pb.plot([y[0,0],y[0,0]],[0,(km.K(y,y)/y/y)[0,0]],'k:')

y = 0.5*np.ones((1,1))
pb.plot(x,km.K(x,y)/x/y)
pb.plot([y[0,0],y[0,0]],[0,(km.K(y,y)/y/y)[0,0]],'k:')

y = 0.75*np.ones((1,1))
pb.plot(x,km.K(x,y)/x/y)
pb.plot([y[0,0],y[0,0]],[0,(km.K(y,y)/y/y)[0,0]],'k:')

pb.ylim((-1,75))

pb.savefig('newfromold-prodfunc-k.pdf',bbox_inches='tight')

########################

K = km.K(x)
Z = np.random.multivariate_normal(0*x[:,0],K,10)

pb.figure(figsize=(5,5))
pb.plot(x,Z[0:5,:].T/x)
pb.ylim((-50,50))

pb.savefig('newfromold-prodfunc-traj.pdf',bbox_inches='tight')



##############################################################################
##############################################################################
## Composition with a function

km = GPy.kern.Matern32(1,1.,1.)
########################
x = np.linspace(0,1,101)[:,None]

fig = pb.figure(figsize=(5,5))

y = 0.1*np.ones((1,1))
pb.plot(x,km.K(1/x,1/y))
pb.plot([y[0,0],y[0,0]],[0,(km.K(1/y,1/y))[0,0]],'k:')

y = 0.25*np.ones((1,1))
pb.plot(x,km.K(1/x,1/y))
pb.plot([y[0,0],y[0,0]],[0,(km.K(1/y,1/y))[0,0]],'k:')

y = 0.5*np.ones((1,1))
pb.plot(x,km.K(1/x,1/y))
pb.plot([y[0,0],y[0,0]],[0,(km.K(1/y,1/y))[0,0]],'k:')

y = 0.75*np.ones((1,1))
pb.plot(x,km.K(1/x,1/y))
pb.plot([y[0,0],y[0,0]],[0,(km.K(1/y,1/y))[0,0]],'k:')

pb.ylim((-0.05,1.05))

pb.savefig('newfromold-compfunc-k.pdf',bbox_inches='tight')

########################
km = GPy.kern.Matern32(1,1.,1.)
x = np.linspace(0.001,1,201)[:,None]

K = km.K(1/x)
Z = np.random.multivariate_normal(0*x[:,0],K,10)

pb.figure(figsize=(5,5))
pb.plot(x,Z[0:5,:].T)

pb.savefig('newfromold-compfunc-traj.pdf',bbox_inches='tight')

