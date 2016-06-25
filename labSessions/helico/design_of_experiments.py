import numpy as np
import pylab as pb
import random
pb.ion()

Wl_bounds = (3,7)
Ww_bounds = (2,5)
Tl_bounds = (5,11)
Al_bounds = (4,14)

U = np.random.uniform(0,1,10000).reshape(10000./4,4)

X = U * np.array([[Wl_bounds[1]-Wl_bounds[0],Ww_bounds[1]-Ww_bounds[0],Tl_bounds[1]-Tl_bounds[0],Al_bounds[1]-Al_bounds[0]]])
X += np.array([[Wl_bounds[0],Ww_bounds[0],Tl_bounds[0],Al_bounds[0]]])

totlength = X[:,0] + X[:,2] + X[:,3]
cosalpha = ((X[:,3]-2.5)**2-(X[:,2]-2.5)**2-X[:,0]**2)/(-2.*(X[:,2]-2.5)*X[:,0])
Xcrop = X[(cosalpha<.4) & (cosalpha>-0.7) & (totlength<27),:]

alphacrop = angle(Xcrop + np.asarray([0,0,-2.5,-2.5])[None,:])

for i in range(alphacrop.shape[0]):    
    X2 = np.zeros((3,2))
    X2[1,:] = Xcrop[i,0]*np.asarray((np.cos(alphacrop[i]),np.sin(alphacrop[i])))
    X2[2,0] = Xcrop[i,2]-2.5
    pb.plot(X2[:,0],X2[:,1],'bx',mew=2)
    
## cluster
#(Ut,ind) = find_centers(box2unit(Xcrop), 30)
#Xt = unit2box(np.asarray(Ut))

(Xt,ind) = find_centers(Xcrop, 30)
Xt = np.asarray(Xt)

data = np.genfromtxt('data.csv',delimiter=',')
Xt = data[:,0:4]

alpha = angle(Xt + np.asarray([0,0,-2.5,-2.5])[None,:])

#np.savetxt("DoE.csv", Xt, delimiter=",")
#make_latex(Xt,'Nicolas')

for i in range(30):    
    X2 = np.zeros((3,2))
    X2[1,:] = Xt[i,0]*np.asarray((np.cos(alpha[i]),np.sin(alpha[i])))
    X2[2,0] = Xt[i,2]-2.5
    pb.plot(X2[:,0],X2[:,1],'g',mew=2)
    
pb.axis([-5,10,-5,10])

def unit2box(U):
    X = U * np.array([[Wl_bounds[1]-Wl_bounds[0],Ww_bounds[1]-Ww_bounds[0],Tl_bounds[1]-Tl_bounds[0],Al_bounds[1]-Al_bounds[0]]])
    X += np.array([[Wl_bounds[0],Ww_bounds[0],Tl_bounds[0],Al_bounds[0]]])
    return(X)


def box2unit(X):
    U = X - np.array([[Wl_bounds[0],Ww_bounds[0],Tl_bounds[0],Al_bounds[0]]])
    U = U / np.array([[Wl_bounds[1]-Wl_bounds[0],Ww_bounds[1]-Ww_bounds[0],Tl_bounds[1]-Tl_bounds[0],Al_bounds[1]-Al_bounds[0]]])
    return(U)


pb.plot(box2unit(Xcrop)[:,0],box2unit(Xcrop)[:,2],'bx',mew=1)
pb.plot(np.asarray(Ut)[:,0],np.asarray(Ut)[:,2],'rx',mew=1)
