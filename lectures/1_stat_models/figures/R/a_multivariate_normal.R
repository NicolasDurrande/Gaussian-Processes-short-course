library(tikzDevice)
library(MASS)

darkPurple <- "#5c3566"
darkBlue <- "#204a87"
darkGreen <- "#4e9a06"
darkChocolate <- "#8f5902"
darkRed  <- "#a40000"
darkOrange <- "#ce5c00"
darkButter <- "#c4a000"


kBrown <- function(x,y,param=1){
  param*outer(x,y,"pmin")
}

kExp <- function(x,y,param=c(1,.2)){
  param[1]*exp(-abs(outer(x,y,"-"))/param[2])
}

kGauss <- function(x,y,param=c(1,.2)){
  param[1]*exp(-.5*(outer(x,y,"-")/param[2])^2)
}

kMat32 <- function(x,y,param=c(1,.2)){
  d <- sqrt(3)*abs(outer(x,y,"-"))/param[2]
  return(param[1]*(1 + d)*exp(-d))
}

kMat52 <- function(x,y,param=c(1,.2)){
  d <- sqrt(5)*abs(outer(x,y,"-"))/param[2]
  return(param[1]*(1 + d + 1/3*d^2)*exp(-d))
}

kWhite <- function(x,y,param=1){
  return(param*outer(x,y,"-")==0)
}


#################################
## 1d
x <- seq(-5,5,length=101)
f <- dnorm(x,0,1)

tikz('MVN_dens1.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(x,f,type='l',lwd=2,col=darkBlue,ylab="density",cex.axis=2,cex.lab=2)
dev.off()
tools::texi2dvi('MVN_dens1.tex',pdf=T)

#################################
## dim d
multdens <- function(x,m,K){
  d <- length(m)
  xc <- matrix(x-m,ncol=1)
  return(1/sqrt((2*pi)^d*det(K)) * exp(-.5*t(xc)%*%solve(K)%*%xc))
}

m <- c(0,0)
K <- matrix(c(3,2,2,3),2)
g <- seq(-5,5,length=31)
G <- as.matrix(expand.grid(g,g))
F <- rep(0,dim(G)[1])
for(i in 1:dim(G)[1]){
  F[i] <- multdens(G[i,],m,K)
}

tikz('MVN_dens2.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(.5,2.1,.5,1.5))
persp(g,g,matrix(F,length(g)),xlab="$x_1$",ylab="$x_2$",zlab="density",cex.axis=2,cex.lab=2,theta = 20, phi = 25)
dev.off()
tools::texi2dvi('MVN_dens2.tex',pdf=T)

#################################
## samples
K <- matrix(c(1,2,2,7),2)
Y <- mvrnorm(700,c(0,2),K)
tikz('MVN_gaussvec1.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(Y,xlab='$Y_1$',ylab='$Y_2$',asp=1,col=rgb(0,0,0,.5),cex.axis=2,cex.lab=2)
dev.off()
tools::texi2dvi('MVN_gaussvec1.tex',pdf=T)

K <- matrix(c(1,0,0,1),2)
Y <- mvrnorm(1000,c(0,0),K)
tikz('MVN_gaussvec2.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(Y,xlab='$Y_1$',ylab='$Y_2$',asp=1,col=rgb(0,0,0,.5),cex.axis=2,cex.lab=2)
dev.off()
tools::texi2dvi('MVN_gaussvec2.tex',pdf=T)

K <- matrix(c(4,-2,-2,1.5),2)
Y <- mvrnorm(1500,c(0,0),K)
for(i in 1:1500){
  if(runif(1)>.7 ) Y[i,1] <- -Y[i,1] 
}
tikz('MVN_gaussvec3.tex', standAlone = TRUE, width=5,height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(Y,xlab='$Y_1$',ylab='$Y_2$',asp=1,col=rgb(0,0,0,.5),cex.axis=2,cex.lab=2)
dev.off()
tools::texi2dvi('MVN_gaussvec3.tex',pdf=T)



##################################################################"
### plot kernel 
x <- seq(from=-5, to=5, length=201)
K <- kMat52(x,x,c(1,50))

tikz('MVN_kern150.tex', standAlone = TRUE, width=5,height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(x,K[,100],type='l',ylab='k(x,0)',cex.axis=1.5,cex.lab=2,ylim=c(0,3))
dev.off()
tools::texi2dvi('MVN_kern150.tex',pdf=T)

m <- 0*x
Z <- mvrnorm(200,m,K)

tikz('MVN_traj150.tex', standAlone = TRUE, width=5,height=5)
plot(x,Z[1,],ylim=c(-6,6),type='l',ylab="samples of Z",cex.axis=1.5,cex.lab=2)
for(i in 2:100){
  lines(x,Z[i,],col=i)
}
dev.off()
tools::texi2dvi('MVN_traj150.tex',pdf=T)


