library(tikzDevice)
library(MASS)
source('a_functions.R')

#################################
## no trend
n <- 5
x <- as.matrix(seq(from=0, to=1, length=201))
X <- as.matrix(seq(from=0, to=0.8, length=n))
ftest <- function(X){
  sin(2*pi*X) + sin(4*pi*X) + 6*X -3
}
F <- ftest(X)

GP <- GPR(x,X,F,k0,c(10,.25))
tikz('exotic_int.tex', standAlone = TRUE, width=8, height=5)
plotGPR(x,GP,"$Z(x)|Z(X)=F,\\int Z = 0$")
points(X, F, pch=4, cex=1,lwd=3)
lines(x,ftest(x))
dev.off()
tools::texi2dvi('exotic_int.tex',pdf=T)


GP <- GPR(x,X,F,kGauss,c(10,.25))
tikz('exotic_pasint.tex', standAlone = TRUE, width=8, height=5)
plotGPR(x,GP,"$Z(x)|Z(X)=F$")
points(X, F, pch=4, cex=1,lwd=3)
lines(x,ftest(x))
dev.off()
tools::texi2dvi('exotic_pasint.tex',pdf=T)

#################################
## with derivatives

x <- as.matrix(seq(from=0, to=5, length=201))
ftest <- function(X){
  sin(2*X) + 2*X - 4
}
dftest <- function(X){
  2*cos(2*X) + 2
}
X <- as.matrix(c(0,1,2,5))
F <- ftest(X)
Xd <- as.matrix(c(2,3,4))
Fd <- dftest(Xd)

par <- c(1,.8)
kx <- t(cbind(kGauss(x,X,par),-dkGauss(x,Xd,par)))
K1 <- cbind(kGauss(X,X,par),-dkGauss(X,Xd,par))
K2 <- cbind(t(-dkGauss(X,Xd,par)),ddkGauss(Xd,Xd,par))
K <- rbind(K1,K2)
Ft <- rbind(F,Fd)

m <- t(kx) %*% solve(K) %*% Ft
v <- kGauss(x,x,par) - t(kx) %*% solve(K) %*% kx
upp95 <- m + 1.96*sqrt(pmax(diag(v),0))
low95 <- m - 1.96*sqrt(pmax(diag(v),0))

GP <- list(m,v,low95,upp95)
tikz('exotic_der.tex', standAlone = TRUE, width=8, height=5)
plotGPR(x,GP,"$Z(x)|Z(X)=F,dZ(X_d)/dx=F_d$")
points(X, F, pch=4, cex=1,lwd=3)
lines(x,ftest(x))
eps <- .2
lines(Xd[1]+c(-eps,eps),m[x==Xd[1]]+Fd[1]*c(-eps,eps),col='red',lwd=3)
lines(Xd[2]+c(-eps,eps),m[x==Xd[2]]+Fd[2]*c(-eps,eps),col='red',lwd=3)
lines(Xd[3]+c(-eps,eps),m[x==Xd[3]]+Fd[3]*c(-eps,eps),col='red',lwd=3)
points(Xd[1],m[x==Xd[1]],pch=16,col='red')
points(Xd[2],m[x==Xd[2]],pch=16,col='red')
points(Xd[3],m[x==Xd[3]],pch=16,col='red')
dev.off()
tools::texi2dvi('exotic_der.tex',pdf=T)

## deriv only

Xd <- as.matrix(c(.5,4))
F <- as.matrix(c(2,0))

par <- c(1,.5)
kx <- t(-dkGauss(x,Xd,par))
K <- ddkGauss(Xd,Xd,par)

m <- t(kx) %*% solve(K) %*% F
v <- kGauss(x,x,par) - t(kx) %*% solve(K) %*% kx
upp95 <- m + 1.96*sqrt(pmax(diag(v),0))
low95 <- m - 1.96*sqrt(pmax(diag(v),0))

GP <- list(m,v,low95,upp95)
tikz('exotic_deronly.tex', standAlone = TRUE, width=8, height=5)
plotGPR(x,GP,"$Z(x)|dZ(X_d)/dx=F_d$")
eps <- .2
lines(Xd[1]+c(-eps,eps),m[x==Xd[1]]+F[1]*c(-eps,eps),col='red',lwd=3)
lines(Xd[2]+c(-eps,eps),m[x==Xd[2]]+F[2]*c(-eps,eps),col='red',lwd=3)
lines(Xd[3]+c(-eps,eps),m[x==Xd[3]]+F[3]*c(-eps,eps),col='red',lwd=3)
points(Xd[1],m[x==Xd[1]],pch=16,col='red')
points(Xd[2],m[x==Xd[2]],pch=16,col='red')
points(Xd[3],m[x==Xd[3]],pch=16,col='red')
dev.off()
tools::texi2dvi('exotic_deronly.tex',pdf=T)

## simulate

Z <- mvrnorm(200,m,v)

tikz('exotic_simul.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plot(x,Z[1,],ylim=c(-3,3),type='l',ylab="samples of cond. GP")
for(i in 2:100){
  lines(x,Z[i,],col=i)
}
dev.off()
tools::texi2dvi('exotic_simul.tex',pdf=T)
