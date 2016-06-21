import numpy as np
import pylab as pb

pb.ion()
pb.close('all')

###############################################
xg = np.linspace(-2,2,201)

def mu(x):
	return(1.*(np.abs(x)<1.))

def sinc(x):
	return(np.sin(x)/x)

pb.figure(figsize=(5,5))
ax = pb.subplot(111)
pb.plot(xg,mu(xg))
pb.ylim((-0.1,1.1))
ax.set_xticklabels(['','','','',0.,'','','',''])
#ax.set_yticklabels(['','','','',0.,'','','',''])
ax.yaxis.set_visible(False)
pb.savefig('Bochner-musinc.pdf',bbox_inches='tight')

y = sinc(2*np.pi*xg)
y[100] = 1.
pb.figure(figsize=(5,5))
ax = pb.subplot(111)
pb.plot(xg,y)
ax.set_xticklabels(['','','','',0.])
ax.yaxis.set_visible(False)
pb.ylim((-0.3,1.1))
pb.savefig('Bochner-ksinc.pdf',bbox_inches='tight')

##################################
## A. Wilson

xg = np.linspace(-10,10,201)

def gauss(x,sig,the,mu):
	return(sig*np.exp(-(x-mu)**2/the) + sig*np.exp(-(x+mu)**2/the))

def gausscos(x,sig,the,mu):
	return(np.exp(-(x)**2/the)*np.cos(x*sig))

###
pb.figure(figsize=(5,5))
ax = pb.subplot(111)
pb.plot(xg,gauss(xg,1.,4.,3.))
pb.ylim((-0.1,1.1))
ax.set_xticklabels(['','',0.,'',''])
#ax.set_yticklabels(['','','','',0.,'','','',''])
ax.yaxis.set_visible(False)
pb.savefig('Bochner-wilsonmu.pdf',bbox_inches='tight')

pb.figure(figsize=(5,5))
ax = pb.subplot(111)
pb.plot(xg,gausscos(xg,1.,4.,3.))
pb.ylim((-0.3,1.1))
ax.set_xticklabels(['','',0.,'',''])
#ax.set_yticklabels(['','','','',0.,'','','',''])
ax.yaxis.set_visible(False)
pb.savefig('Bochner-wilsonk.pdf',bbox_inches='tight')


###
pb.figure(figsize=(5,5))
ax = pb.subplot(111)
pb.plot(xg,gauss(xg,1.,4.,6.)+gauss(xg,.3,.01,1.)+gauss(xg,2,.2,5.)+gauss(xg,5.,6.,2.))
#pb.ylim((-0.1,1.1))
ax.set_xticklabels(['','',0.,'',''])
#ax.set_yticklabels(['','','','',0.,'','','',''])
ax.yaxis.set_visible(False)
pb.savefig('Bochner-wilsonmus.pdf',bbox_inches='tight')

+gausscos(xg,1,.05,1.)+gausscos(xg,2,.1,5.)

Kv = gausscos(xg,1.,4.,6.)+gausscos(xg,5.,6.,2.) + gausscos(xg,4,1.,10.)

pb.figure(figsize=(5,5))
ax = pb.subplot(111)
pb.plot(xg,Kv)
#pb.ylim((-0.3,1.1))
ax.set_xticklabels(['','',0.,'',''])
#ax.set_yticklabels(['','','','',0.,'','','',''])
ax.yaxis.set_visible(False)
pb.savefig('Bochner-wilsonks.pdf',bbox_inches='tight')

K = np.zeros((100,100))
for i in range(100):
	K[i,:] = Kv[100-i:200-i]

Z = np.random.multivariate_normal([0]*100,K,3)

pb.figure(figsize=(8,5))
pb.plot(np.linspace(0,5,100).T,Z.T)
pb.savefig('Bochner-wilsonktraj.pdf',bbox_inches='tight')
