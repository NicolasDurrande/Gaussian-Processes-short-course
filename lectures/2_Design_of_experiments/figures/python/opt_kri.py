import numpy as np
import pylab as pb

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

#########################

X = np.vstack(((0,0),(0,.5),(0,1),(.5,0),(.5,.5),(.5,1),(1,0),(1,.5),(1,1)))
g = np.linspace(0,0.99,100)+0.05
G = np.zeros((100**2,2))
for i in range(100):
	for j in range(100):
		G[100*i+j,:] = (g[i],g[j])


def kGauss(X,Y):
	d2 = np.sum((X[:,None,:]-Y[None,:,:])**2,2)
	k = np.exp(-d2/2/0.2**2)
	return(k)

eps = np.zeros((9,2))
eps[4,1] = 0.01

np.linalg.det(kGauss(X,X))
np.linalg.det(kGauss(X+eps,X+eps))
(np.linalg.det(kGauss(X+eps,X+eps))-np.linalg.det(kGauss(X,X)))/0.01

pb.figure(figsize=(4,4))
pb.plot(X[:,0],X[:,1],'x',markersize=15,mew=3,color=tango[2])
pb.xlabel('$x_1$')
pb.ylabel('$x_2$')

pb.savefig('opt_XD.pdf',bbox_inches='tight')

## Shrink DoE
S = np.linspace(.5,1,50)
Iopt = 0*S
Gopt = 0*S
for i,s in enumerate(S):
	Xs = (X-.5)*s + .5
	VP = 1 - np.sum(np.dot(np.linalg.inv(kGauss(Xs,Xs)),kGauss(Xs,G))* kGauss(Xs,G),axis=0)
	Iopt[i] = np.mean(VP)
	Gopt[i] = np.max(VP)


pb.figure(figsize=(5,3))
pb.plot(S,Iopt,linewidth=2,color=tango[0])
pb.plot(S,Gopt,linewidth=2,color=tango[1])
pb.xlim((.45,1.05))
pb.ylim((0,1.7))
pb.xlabel('shrinking factor')
pb.ylabel('criteria')
pb.legend(('I-optimality','G-optimality'))

pb.savefig('opt_IG.pdf',bbox_inches='tight')


XI = (X-.5)*S[np.argmin(Iopt)] + .5
XG = (X-.5)*S[np.argmin(Gopt)] + .5

pb.figure(figsize=(4,4))
pb.plot(XI[:,0],XI[:,1],'x',markersize=15,mew=3,color=tango[0])
pb.plot(XG[:,0],XG[:,1],'x',markersize=15,mew=3,color=tango[1])
pb.xlabel('$x_1$')
pb.ylabel('$x_2$')

pb.savefig('opt_XIG.pdf',bbox_inches='tight')

