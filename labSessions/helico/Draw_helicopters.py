import numpy as np
import pylab as pb
import random

def single_helico_str(X,expNumber,groupName):
	(Wl, Ww, Tl, Al) = X
	Aw = .7
	Tb = 1.
	Tw = 1.2
	lineHeight = .5

	helico_str = """\\makebox[%fcm]{
	\\begin{pspicture}(%f,%f)(%f,%f)
	\psline[linewidth=0.02](%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)
	\psline[linewidth=0.02](%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)(%f,%f)
	\psline[linewidth=0.02](%f,%f)(%f,%f)
	\psline[linewidth=0.02, linestyle=dashed, dash=0.17cm 0.10cm](%f,%f)(%f,%f)
	\psline[linewidth=0.02, linestyle=dashed, dash=0.17cm 0.10cm](%f,%f)(%f,%f)
	\psline[linewidth=0.02, linestyle=dashed, dash=0.17cm 0.10cm](%f,%f)(%f,%f)
	\\rput[b](0,%f){\\footnotesize{%s}}
	\\rput[b](0,%f){\\footnotesize{exp %i}}
	\\rput[b](0,%f){\\footnotesize{$W_l = %.2f$}}
	\\rput[b](0,%f){\\footnotesize{$W_w = %.2f$ }}
	\\rput[b](0,%f){\\footnotesize{$T_l = %.2f$}}
	\\rput[b](0,%f){\\footnotesize{$A_l= %.2f$  }}
	\end{pspicture}
	}%%
	""" %(2*Ww,
		Ww,Wl+Al,-Ww,-Tl,
		0,Wl+Al, -Aw,Wl+Al, -Aw,Wl, -Ww,Wl, -Ww,0, -Tw,-Tb, -Tw,-Tl, 0,-Tl,
		0,Wl+Al,  Aw,Wl+Al,  Aw,Wl,  Ww,Wl,  Ww,0,  Tw,-Tb,  Tw,-Tl, 0,-Tl,
		0,Wl+Al,0,0,
		-Aw,Wl,Aw,Wl,
		-Ww,0,Ww,0,
		-Aw,Wl+Al-2.5,Aw,Wl+Al-2.5,
		-Tb-1,groupName,
		-Tb-1-lineHeight,expNumber,
		-Tb-1-2*lineHeight,Wl,
		-Tb-1-3*lineHeight,Ww,
		-Tb-1-4*lineHeight,Tl,
		-Tb-1-5*lineHeight,Al)
	
	return(helico_str)


def make_latex(X,groupName):
	f = open('helicopters.tex', 'w')

	f.write( """\documentclass{article}
	\usepackage[usenames,dvipsnames]{pstricks}
	\usepackage[top=5mm,bottom=5mm,right=5mm,left=5mm]{geometry}
	\\begin{document} 
	\\raggedbottom 
	""")
	
	wleft = 19.5
	for i in range(X.shape[0]):
		if 2*X[i,1] < wleft:
			f.write(single_helico_str(X[i,:],i+1,groupName))
		else:
			wleft = 19.5
			f.write("\n \n" + single_helico_str(X[i,:],i+1,groupName))
		wleft -= 2*X[i,1]

	f.write('\end{document}')
	f.close()

## Example
#import numpy as np
#X = np.array([[2,2,4,8],[6,5,8,12]]) 
#make_latex(X,'Group Name')


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

def angle(X):
    return(np.arccos(-1.*(X[:,3]**2-X[:,2]**2-X[:,0]**2)/(2*X[:,2]*X[:,0])))
