import numpy as np
import pylab as pb
from scipy.stats import norm
import GPy
pb.ion()

def ftest(x):
	return(-np.sin(x*12)+x*1.2+2)

def ftestn(x):
	return(-np.sin(x*12)+x*1.2+2+np.random.normal(0,.1,x.shape))


# load data
X = np.array([[0,.25,.5,.66,.9]]).T
#X = np.linspace(0,1,10)[:,None]
F = ftest(X)
pb.plot(X,F,'kx')
d = X.shape[1]

# define noise variance
tau2 = 0

# define a kernel
kern = GPy.kern.Matern52(input_dim=d,variance=10,lengthscale=.4)

# define a model
m = GPy.models.gp_regression.GPRegression(X, F, kern)
m['.*noise'].fix(tau2)		# fix the noise variance
print m

m.plot(plot_limits=[0,1])
#pb.savefig('ego_0.pdf',bbox_inches='tight')

m.plot(plot_limits=[0,1])
pb.plot([0,1],[np.min(F)]*2,'k--',linewidth=1.5)
pb.savefig('ego_improv.pdf',bbox_inches='tight')

#####################################################
# proba of improvement
x = np.linspace(0,1,100)[:,None]  
mean, var = m.predict(x)

PI = norm.cdf(np.min(F),mean,np.sqrt(var))
m.plot(plot_limits=[0,1])
pb.plot([0,1],[np.min(F)]*2,'k--',linewidth=1.5)
pb.plot(x,PI,'r',linewidth=2)
pb.ylim((-.2,6))
#pb.savefig('ego_PI.pdf',bbox_inches='tight')

#####################################################
# EI
def EI(x,m):
	mean, var = m.predict(x)
	var[var<0] = 0
	u = (np.min(m.Y) - mean)/(np.sqrt(var))
	ei = np.sqrt(var) * (u * norm.cdf(u) + norm.pdf(u))
	ei[np.isnan(ei)] = 0
	return(ei)

x = np.linspace(0,1,100)[:,None]  
ei = EI(x)

m.plot(plot_limits=[0,1])
pb.plot([0,1],[np.min(F)]*2,'k--',linewidth=1.5)
pb.plot(x,ei*40,'r',linewidth=2)
pb.ylim((-.2,6))
#pb.savefig('ego_EI.pdf',bbox_inches='tight')

## llop
x = np.linspace(0,1,500)[:,None]  
X = np.linspace(0,.9,4)[:,None]
F = ftest(X)

for i in range(6):
	m = GPy.models.gp_regression.GPRegression(X, F, kern)
	m['.*nois'] = 0
	ei = EI(x,m)
	xstar = x[np.argmax(ei)]
	m.plot(plot_limits=[0,1])
	pb.plot([0,1],[np.min(F)]*2,'k--',linewidth=1.5)
	pb.plot(x,ei*20,'r',linewidth=2)
	pb.ylim((-.1,6))
	X = np.vstack((X,xstar))
	F = np.vstack((F,ftest(xstar)))
	pb.plot([xstar]*2,[0,max(ei)*20],'r--',linewidth=1.5)
	pb.savefig('ego_EI%i.pdf'%i,bbox_inches='tight')



K = m.kern.K(X,X)
lamb = np.linalg.eigvals(K)
round(lamb,2)
np.linalg.cond(K)


##############
## osborn taylor

X = np.linspace(0.5,0.51,2)[:,None]
K = m.kern.K(X,X)
np.linalg.cond(K)
np.linalg.eigvals(K)

Kd = K.copy()
Kd[0,1] = (K[0,1]-K[0,0])/.01
Kd[1,0] = (K[0,1]-K[0,0])/.01
Kd[1,1] = (K[1,1]+K[0,0]-2*K[0,1])/.0001
np.linalg.cond(Kd)
np.linalg.eigvals(Kd)

pb.figure(figsize=(5,5))
pb.plot(X,ftest(X),'kx',mew=1.5)
pb.xlim((0,1))
pb.ylim((0,6))
pb.savefig('osborn0',bbox_inches='tight')

pb.figure(figsize=(5,5))
pb.plot(X[0],ftest(X[0]),'kx',mew=1.5)
p = (ftest(X[1]) - ftest(X[0]))/.01
D = .05
pb.plot([X[0]-D,X[0]+D],[ftest(X[0])-p*D,ftest(X[0])+p*D],'r',linewidth=1.5)
pb.xlim((0,1))
pb.ylim((0,6))
pb.savefig('osborn1',bbox_inches='tight')

#####################################################
# noisy EI

def EIn1(x,m):
	mX, vX = m.predict(m.X)
	mn = GPy.models.gp_regression.GPRegression(X, mX, kern)
	mn['.*nois'] = 0.0
	mean, var = mn.predict(x)
	var[var<0] = np.inf
	u = (np.min(mX) - mean)/(np.sqrt(var))
	ei = np.sqrt(var) * (u * norm.cdf(u) + norm.pdf(u))
	return(ei)

def EMI(x,m):
	mean, var = m.predict(x)
	mX, vX = m.predict(m.X)
	var[var<0] = np.inf
	u = (np.min(mX) - mean)/np.sqrt(var+m['.*noise'])
	sig2 = var/(var+m['.*noise'])
	ein = np.sqrt(var+m['.*noise']) * (u*norm.cdf(u) + sig2*norm.pdf(u))
	return(ein)


x = np.linspace(0,1,100)[:,None]  
ei = EMI(x,m)

## llop
x = np.linspace(0,1,500)[:,None]  
X = np.linspace(0,.9,4)[:,None]
F = ftestn(X)

for i in range(6):
	m = GPy.models.gp_regression.GPRegression(X, F, kern)
	m['.*nois'] = 0.01
	ei = EIn1(x,m)
	xstar = x[np.argmax(ei)]
	mX, vX = m.predict(m.X)
	mn = GPy.models.gp_regression.GPRegression(X, mX, kern)
	mn['.*nois'] = 0.0
	pb.figure(figsize=(8,5))
	ax=pb.subplot(111)
	m.plot(plot_limits=[0,1],ax=ax)
	mn.plot(plot_limits=[0,1],ax=ax,linecol='g', fillcol='g')
	pb.plot(x,ei*20,'r',linewidth=2)
	pb.ylim((-.1,6))
	X = np.vstack((X,xstar))
	F = np.vstack((F,ftestn(xstar)))
	pb.plot([xstar]*2,[0,max(ei)*20],'r--',linewidth=1.5)
	pb.savefig('ego_EI1n%i.pdf'%i,bbox_inches='tight')


for i in range(12):
	m = GPy.models.gp_regression.GPRegression(X, F, kern)
	m['.*nois'] = 0.01
	ei = EMI(x,m)
	xstar = x[np.argmax(ei)]
	mX, vX = m.predict(m.X)
	pb.figure(figsize=(8,5))
	ax=pb.subplot(111)
	m.plot(plot_limits=[0,1],ax=ax)
	pb.plot(x,ei*20,'r',linewidth=2)
	pb.ylim((-.1,6))
	X = np.vstack((X,xstar))
	F = np.vstack((F,ftestn(xstar)))
	pb.plot([xstar]*2,[0,max(ei)*20],'r--',linewidth=1.5)
	# pb.savefig('ego_EI1n%i.pdf'%i,bbox_inches='tight')


m.predict(np.array([[.13]]))

ones = np.ones((20,20))

ones - np.dot(np.dot(ones,np.linalg.inv(ones+np.eye(20))),ones)

#####################################################
# inversion

# load data
X = np.array([[0,.25,.5,.66,.9]]).T
F = ftest(X)
pb.plot(X,F,'kx')
d = X.shape[1]

# define noise variance
tau2 = 0

# define a kernel
kern = GPy.kern.Matern52(input_dim=d,variance=10,lengthscale=.4)

# define a model
m = GPy.models.gp_regression.GPRegression(X, F, kern)
m['.*noise'].fix(tau2)		# fix the noise variance

m.plot(plot_limits=[0,1])
pb.plot([0,1],[3.2]*2,'k--',linewidth=1.5)
pb.savefig('inv.pdf',bbox_inches='tight')


##
mean, var = m.predict(x)
prob = norm.pdf(3.2,mean,var)
m.plot(plot_limits=[0,1])
pb.plot([0,1],[3.2]*2,'k--',linewidth=1.5)
pb.plot(x,prob)
pb.ylim((-.1,6))
pb.savefig('invproba.pdf',bbox_inches='tight')


xstar = x[np.argmax(prob)]
X = np.vstack((X,xstar))
F = np.vstack((F,ftestn(xstar)))
m = GPy.models.gp_regression.GPRegression(X, F, kern)
m['.*noise'].fix(tau2)		# fix the noise variance

mean, var = m.predict(x)
prob = norm.pdf(3.2,mean,var)
m.plot(plot_limits=[0,1])
pb.plot([0,1],[3.2]*2,'k--',linewidth=1.5)
pb.plot(x,prob)
pb.ylim((-.1,6))
pb.savefig('invproba1.pdf',bbox_inches='tight')


xstar = x[np.argmax(prob[:250])]
X = np.vstack((X,xstar))
F = np.vstack((F,ftestn(xstar)))
m = GPy.models.gp_regression.GPRegression(X, F, kern)
m['.*noise'].fix(tau2)		# fix the noise variance

mean, var = m.predict(x)
prob = norm.pdf(3.2,mean,var)
m.plot(plot_limits=[0,1])
pb.plot([0,1],[3.2]*2,'k--',linewidth=1.5)
pb.plot(x,prob)
pb.ylim((-.1,6))
pb.savefig('invproba2.pdf',bbox_inches='tight')


xstar = x[np.argmax(prob[:250])]
X = np.vstack((X,xstar))
F = np.vstack((F,ftestn(xstar)))
m = GPy.models.gp_regression.GPRegression(X, F, kern)
m['.*noise'].fix(tau2)		# fix the noise variance

mean, var = m.predict(x)
prob = norm.pdf(3.2,mean,var)
m.plot(plot_limits=[0,1])
pb.plot([0,1],[3.2]*2,'k--',linewidth=1.5)
pb.plot(x,prob)
pb.ylim((-.1,6))
pb.savefig('invproba3.pdf',bbox_inches='tight')


