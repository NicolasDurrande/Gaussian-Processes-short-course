library(tikzDevice)
library(MASS)
source('a_functions.R')

##################################################################"
### linear regression

## points
n <- 10
x <- seq(from=0, to=1, length=101)
X <- seq(from=.05, to=.95, length=n)
F <- X^2 - .5*X + 1 + rnorm(length(X),0,.05)
f <- x^2 - .5*x + 1
H <- function(x){
  cbind(0*x+1,x,x^2)
}

beta <- solve(t(H(X))%*%H(X)) %*% t(H(X)) %*% F
m <- H(x) %*% beta
varpred <- H(x) %*% solve(t(H(X))%*%H(X)) %*% t(H(X)) %*% diag(rep(0.05^2,n)) %*% H(X) %*% t(solve(t(H(X))%*%H(X))) %*% t(H(x))
low95 <- m - 1.96*sqrt(pmax(diag(varpred)))
upp95 <- m + 1.96*sqrt(pmax(diag(varpred)))
GP <- list(m,var,low95,upp95)

# data points
tikz('linreg_0.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plot(X, F, pch=4, cex=1,lwd=3,xlab='$x$', ylab='$F$', ylim=range(low95-0.5,upp95+0.5))
dev.off()
tools::texi2dvi('linreg_0.tex',pdf=T)

# basis functions
tikz('linreg_1.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plot(X, F, pch=4,type="n", cex=1,lwd=3,xlab='$x$', ylab='$B(x)$', ylim=range(0,1))
lines(x,H(x)[,1],col=darkGreen,lwd=3)
lines(x,H(x)[,2],col=darkChocolate,lwd=3)
lines(x,H(x)[,3],col=darkRed,lwd=3)
dev.off()
tools::texi2dvi('linreg_1.tex',pdf=T)

# model
tikz('linreg_2.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plot(X, F, pch=4,type="n", cex=1,lwd=3,xlab='$x$', ylab='$m(x)$', ylim=range(low95-0.5,upp95+0.5))
lines(x,m,col=darkBlue,lwd=3)
points(X, F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('linreg_2.tex',pdf=T)

# model error
tikz('linreg_3.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plot(X, F, pch=4,type="n", cex=1,lwd=3,xlab='$x$', ylab='$m(x)$ and $f(x)$', ylim=range(low95-0.5,upp95+0.5))
lines(x,m,col=darkBlue,lwd=3)
lines(x,f,col=darkRed,lwd=3)
points(X, F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('linreg_3.tex',pdf=T)

# model uncertainty
K = solve(t(H(X))%*%H(X)) %*% t(H(X)) %*% diag(rep(0.05^2,n)) %*% H(X) %*% t(solve(t(H(X))%*%H(X)))
BetaHat = mvrnorm(1000,beta,K)

tikz('linreg_4.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plot(X, F, pch=4,type="n", cex=1,lwd=3,xlab='$x$', ylab='samples from $m(x)$', ylim=range(low95-0.5,upp95+0.5))
for(i in 1:10) lines(x,H(x) %*% matrix(BetaHat[i,],ncol=1),col=darkBlue,lwd=3)
dev.off()
tools::texi2dvi('linreg_4.tex',pdf=T)

## Confidence intervals
tikz('linreg_5.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plotGPR(x,GP, ylab='linear regression')
points(X, F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('linreg_5.tex',pdf=T)

##########
# minimizing 

xstar <- -beta[2]/beta[3]/2
Xstar <- -BetaHat[,2]/BetaHat[,3]/2

Xstar <- Xstar[Xstar<1 & Xstar>0]
aa <- hist(Xstar,30,plot=FALSE)
aa$density <- aa$density/10

## Confidence intervals
tikz('linreg_6.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,5.1),cex.axis=1.5,cex.lab=2)
plotGPR(x,GP, ylab='model',ylim=c(0,1.8))
points(X, F, pch=4, cex=1,lwd=3)

plot(aa,axes=F,freq=FALSE,ylim=c(0,30),main='',xlab='',ylab='',add=T)
axis(4,at = c(0,.5,1,1.5),labels=c(0,5,10,15), lwd=2)
mtext(4,text="Density of opt. input",line=3,cex=2)
lines(c(0,1),c(0,0))
dev.off()
tools::texi2dvi('linreg_6.tex',pdf=T)

