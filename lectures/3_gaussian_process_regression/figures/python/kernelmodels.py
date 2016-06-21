import numpy as np
import pylab as pb
import GPy

pb.ion()
pb.close('all')

###############################################
xg = np.linspace(0,1,101)

def ftest(x):
	return(np.sin(2*np.pi*x)+ 2*x)

X = np.linspace(.1,.9,5)[:,None]
Y = ftest(X)

pb.plot(X,Y,'kx',mew=1.5)
pb.xlim((0.,1.))
pb.xlim((-.1,1.1))

k1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=1.5)
m1 = GPy.models.GPRegression(X,Y,k1)
m1.ensure_default_constraints()
m1.constrain_fixed('noise',0.)
m1.optimize()
pb.figure(figsize=(5,5))
ax = pb.subplot(111)
m1.plot(ax=ax)
pb.savefig('kernelmodels-inf1.pdf',bbox_inches='tight')

k2 = GPy.kern.exponential(input_dim=1,variance=1.,lengthscale=1.5)
m2 = GPy.models.GPRegression(X,Y,k2)
m2.ensure_default_constraints()
m2.constrain_fixed('noise',0.)
m2.optimize()
pb.figure(figsize=(5,5))
ax = pb.subplot(111)
m2.plot(ax=ax)
pb.savefig('kernelmodels-inf2.pdf',bbox_inches='tight')

########################################
## additive vs product test function
def ftest(X):
	return(np.sin(4*np.pi*X[:,0]) + np.cos(4*np.pi*X[:,1]) + 2*X[:,1] )

xg = np.linspace(0,1,31)
XX,YY = np.meshgrid(xg,xg)

kg1 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)
kg2 = GPy.kern.rbf(input_dim=1,variance=1.,lengthscale=.15)

X = np.random.uniform(0,1,(6,2))
Y = ftest(X)[:,None]

Xg,Yg = np.meshgrid(xg,xg)
XX = np.hstack((Xg.flatten()[:,None],Yg.flatten()[:,None]))

Yt = ftest(XX)[:,None]

#####################
# add
# kernel
ka = kg1.add(kg2,tensor=True)
# model
ma = GPy.models.GPRegression(X,Y,ka)
ma['noise'] = 0.

# predict 
Ya,Va,_,_ = ma.predict(XX)
RMSEa = np.sqrt(np.mean((Ya-Yt)**2))

# plot
pb.figure(figsize=(5,5))
cont = pb.contour(Xg,Yg,Ya.reshape((31,31)))
#pb.clabel(cont, inline=1, fontsize=10)
pb.plot(X[:,0],X[:,1],'kx',mew=1.5)
pb.xlabel('$x_1$', fontsize=20)
pb.ylabel('$x_2$', fontsize=20)
pb.savefig('kernelmodels-predm.pdf',bbox_inches='tight')


# plot
pb.figure(figsize=(5,5))
cont = pb.contour(Xg,Yg,Va.reshape((31,31)))
#pb.clabel(cont, inline=1, fontsize=10)
pb.plot(X[:,0],X[:,1],'kx',mew=1.5)
pb.xlabel('$x_1$', fontsize=20)
pb.ylabel('$x_2$', fontsize=20)
pb.savefig('kernelmodels-predvar.pdf',bbox_inches='tight')


#####################
# prod
# kernel
ka = kg1.prod(kg2,tensor=True)
# model
ma = GPy.models.GPRegression(X,Y,ka)
ma['noise'] = 0.

# predict 
Ya,Va,_,_ = ma.predict(XX)
RMSEa = np.sqrt(np.mean((Ya-Yt)**2))

# plot
pb.figure(figsize=(5,5))
cont = pb.contour(Xg,Yg,Va.reshape((31,31)))
#pb.clabel(cont, inline=1, fontsize=10)
pb.plot(X[:,0],X[:,1],'kx',mew=1.5)
pb.xlabel('$x_1$', fontsize=20)
pb.ylabel('$x_2$', fontsize=20)
pb.savefig('kernelmodels-predvarprod.pdf',bbox_inches='tight')


###############
## plan axes
xp = np.linspace(0.05,0.95,5)[:,None]
ax1 = np.hstack((xp,0*xp+0.05))
ax2 = np.hstack((xp*0+0.05,xp))

X = np.vstack(([0.05,0.05],ax1,ax2))
Y = ftest(X)[:,None]

xg = np.linspace(0,1,31)
Xg,Yg = np.meshgrid(xg,xg)
XX = np.hstack((Xg.flatten()[:,None],Yg.flatten()[:,None]))

# kernel
ka = kg1.add(kg2,tensor=True)
# model
ma = GPy.models.GPRegression(X,Y,ka)
ma['noise'] = 0.

# predict 
Ya,Va,_,_ = ma.predict(XX)
RMSEa = np.sqrt(np.mean((Ya-Yt)**2))

# plot
pb.figure(figsize=(5,5))
cont = pb.contour(Xg,Yg,Va.reshape((31,31)))
#pb.clabel(cont, inline=1, fontsize=10)
pb.plot(X[:,0],X[:,1],'kx',mew=1.5)
pb.xlabel('$x_1$', fontsize=20)
pb.ylabel('$x_2$', fontsize=20)
pb.savefig('kernelmodels-predaxe.pdf',bbox_inches='tight')


