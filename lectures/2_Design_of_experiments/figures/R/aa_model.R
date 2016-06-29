library(DiceKriging)
library(tikzDevice)
library(MASS)
source('aa_functions.R')

# lightblue <- rgb(114/255,159/255,207/255,.3)
# darkblue <- rgb(32/255,74/255,135/255,1)
# darkbluetr <- rgb(32/255,74/255,135/255,.3)
# 
# plotGPR <- function(x,m,low95,upp95,ylab="$Z(x)$"){
#   par(mar=c(4.5,5.1,1.5,1.5))
#   plot(x, m, type="n", xlab="$x$",ylab=ylab, ylim=range(low95-0.5,upp95+0.5),cex.axis=2,cex.lab=2)
#   polygon(c(x,rev(x)),c(upp95,rev(low95)),border=NA,col=lightblue)
#   lines(x,m,col=darkblue,lwd=3)
#   lines(x,low95,col=darkbluetr)  
#   lines(x,upp95,col=darkbluetr)  
# }
# 
# kBrown <- function(x,y,sig2=1){
#   sig2*outer(x,y,"pmin")
# }
# 
# kExp <- function(x,y,sig2=1,theta=1){
#   sig2*exp(-abs(outer(x,y,"-"))/theta)
# }
# 
# kGauss <- function(x,y,sig2=1,theta=.2){
#   sig2*exp(-.5*(outer(x,y,"-")/theta)^2)
# }
# 
# kMat32 <- function(x,y,sig2=1,theta=.2){
#   d <- sqrt(3)*abs(outer(x,y,"-"))/theta
#   return(sig2*(1 + d)*exp(-d))
# }
# 
# kWhite <- function(x,y,sig2=1){
#   return(sig2*as.numeric(outer(x,y,"-")==0))
# }

##################################################################"
##################################################################"
### Modeles de krigeage sans bruit
n <- 5
x <- matrix(seq(from=0, to=1, length=101),ncol=1)
X <-  matrix(seq(from=0.1, to=0.9, length=n),ncol=1)
#X <-  X + c(0.2,-.1,.1,.25,.1)
f <- function(X) {sin(2*pi*X) + X}
F <- f(X)

GPC <- GPR(x,X,F,kGauss,param=c(1,.4),kernNoise=kWhite,paramNoise=0.001)

tikz('model_0.tex', standAlone = TRUE, width=8, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plotGPR(x,GPC,"model")
#lines(x,f(x),lty=2,col='red',lwd=1.5)
points(X, F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('model_0.tex',pdf=T)


tikz('model_1.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
plotGPR(x,GPC,"model")
points(X, F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('model_1.tex',pdf=T)

sum((GPC[[1]]-f(x))^2)
mean(diag(GPC[[2]]))

##################################################################"
##################################################################"
### GPR = linear combination of basis functions GAUSS
n <- 5
m <- 101
x <- seq(from=-0.5, to=1.5, length=m)

X <- seq(from=0.1, to=0.9, length=n)
F <- c(0.5, 0, 1.5, 3, 2)

GPR <- function(x,X,F,kern,param,kernNoise=kWhite,paramNoise=0)
  
m <- kGauss(x,X,1,.2)%*%solve(kGauss(X,X,1,.2))%*%F
K <- kGauss(x,x,1,.2) - kGauss(x,X,1,.2)%*%solve(kGauss(X,X,1,.2))%*%kGauss(X,x,1,.2)

upp95 <- m + 2*sqrt(pmax(0,diag(K)))
low95 <- m - 2*sqrt(pmax(0,diag(K)))

tikz('ch34_GPRbasisfuncGauss.tex', standAlone = TRUE, width=7, height=5)
plotGPR(x,m,low95,upp95,"$Z(x)|Z(X)=F$")
points(X, F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('ch34_GPRbasisfuncGauss.tex',pdf=T)


tikz('ch34_basisfuncGauss.tex', standAlone = TRUE, width=7, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(x,kGauss(X,x,1,.2)[1,],type="l",col=darkblue,ylab="$k(x,X)$",cex.axis=2,cex.lab=2)
for(i in 2:n) lines(x,kGauss(X,x,1,.2)[i,],col=darkblue)
dev.off()
tools::texi2dvi('ch34_basisfuncGauss.tex',pdf=T)


##################################################################"
##################################################################"
### GPR = linear combination of basis functions Brown
n <- 5
m <- 101
x <- seq(from=0, to=1, length.out=m)

X <- seq(from=0.1, to=0.9, length=n)
F <- c(0.5, 0, 1.5, 3, 2)

m <- kBrown(x,X)%*%solve(kBrown(X,X))%*%F
K <- kBrown(x,x) - kBrown(x,X)%*%solve(kBrown(X,X))%*%kBrown(X,x)

upp95 <- m + 2*sqrt(pmax(0,diag(K)))
low95 <- m - 2*sqrt(pmax(0,diag(K)))

tikz('ch34_GPRbasisfuncBrown.tex', standAlone = TRUE, width=7, height=5)
plotGPR(x,m,low95,upp95,"$Z(x)|Z(X)=F$")
points(X, F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('ch34_GPRbasisfuncBrown.tex',pdf=T)


tikz('ch34_basisfuncBrown.tex', standAlone = TRUE, width=7, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(x,kBrown(X,x)[n,],type="l",col=darkblue,ylab="$k(x,X)$",cex.axis=2,cex.lab=2)
for(i in 1:(n-1)) lines(x,kBrown(X,x,1,.2)[i,],col=darkblue)
dev.off()
tools::texi2dvi('ch34_basisfuncBrown.tex',pdf=T)



##################################################################"
##################################################################"
### Modeles de krigeage avec bruit
n <- 5
m <- 101
x <- seq(from=0, to=1, length=m)

X <- seq(from=0.1, to=0.9, length=n)
F <- c(0.5, 0, 1.5, 3, 2)

m <- kGauss(x,X)%*%solve(kGauss(X,X)+kWhite(X,X,0.1))%*%F
K <- kGauss(x,x) - kGauss(x,X)%*%solve(kGauss(X,X)+kWhite(X,X,0.1))%*%kGauss(X,x)

upp95 <- m + 2*sqrt(pmax(0,diag(K)))
low95 <- m - 2*sqrt(pmax(0,diag(K)))

tikz('ch34_GPRnoise01.tex', standAlone = TRUE, width=5, height=5)
plotGPR(x,m,low95,upp95,"$Z(x)|Z(X)+N(X)=F$")
points(X, F, pch=4, cex=1,lwd=3,col=rgb(0,0,0,.3))
dev.off()
tools::texi2dvi('ch34_GPRnoise01.tex',pdf=T)


##################################################################"
##################################################################"
### Chap 4 parameter estimation
source('aa_functions.R')

## likelihood
x <- seq(0,5,length=50)
e1 <- rexp(20,1)

l2 <- signif(prod(dexp(e1,2)),2)
l1 <- signif(prod(dexp(e1,1)),2)
l05 <- signif(prod(dexp(e1,.5)),2)

tikz('ch34_likelihood.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(x,dexp(x,2),type="l",lwd=2,col=tangocol[1],ylab="density",cex.axis=2,cex.lab=2)
lines(x,dexp(x,1),lwd=2,col=tangocol[2])
lines(x,dexp(x,.5),lwd=2,col=tangocol[3])
points(e1, 0*e1, pch=4, cex=1,lwd=3)
#legend('topright', lty=c(1,1,1),lwd=2,col=tangocol[1:3],legend=c(paste('$\\lambda = 2,\ L=',l2,'$'),paste('$\\lambda = 1,\ L=',l1,'$'),paste('$\\lambda = 0.5,\ L=',l05,'$')),cex=1.5 )
legend('topright', lty=c(1,1,1),lwd=2,col=tangocol[1:3],legend=c('$\\lambda = 2,\ L=4.9 \\ 10^{-14}$','$\\lambda = 1,\ L=2.2 \\ 10^{-10}$','$\\lambda = 0.5,\ L=1.4 \\ 10^{-11}$'),cex=1.5 )
dev.off()
tools::texi2dvi('ch34_likelihood.tex',pdf=T)

## likelihood graph
lambda <- seq(0,5,length=150)
L <- 0*lambda
lL <- 0*lambda
for(i in 1:length(lambda)){
  L[i] <- prod(dexp(e1,lambda[i]))
  lL[i] <- sum(dexp(e1,lambda[i],log=T))
}

tikz('ch34_likelihoodgraph.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(lambda,L,type="l",xlab='$\\lambda$',lwd=2,col=tangocol[1],ylab="likelihood",cex.axis=2,cex.lab=2)
dev.off()
tools::texi2dvi('ch34_likelihoodgraph.tex',pdf=T)

tikz('ch34_loglikelihoodgraph.tex', standAlone = TRUE, width=5, height=5)
par(mar=c(4.5,5.1,1.5,1.5))
plot(lambda,lL,type="l",xlab='$\\lambda$',lwd=2,col=tangocol[1],ylab="log likelihood",cex.axis=2,cex.lab=2)
dev.off()
tools::texi2dvi('ch34_loglikelihoodgraph.tex',pdf=T)


sort(lL, index.return=T)
lambda[28]























###############################################
### FIG 8 le mod?le revient vers la moyenne lorsque on est loin des observations
n <- 5
m <- 501
design <- seq(from=0, to=1, length=n)
response <- c(0.5, 0, 1.5, 3, 2)

covtype <- "gauss"
coef.cov <- 0.2
sigma <- 1.5

trend <- c(intercept <- 0)
model <- km(formula=~1, design=data.frame(x=design), response=response, 
            covtype=covtype, coef.trend=trend, coef.cov=coef.cov, 
            coef.var=sigma^2)

newdata <- seq(from=-2, to=3, length=m)
nsim <- 40
y <- simulate(model, nsim=nsim, newdata=newdata, cond=TRUE, nugget.sim=1e-5)

p <- predict(model, newdata=newdata, type="SK")

plot(design, response, pch=19, cex=0.5, col="black",ylim=range(y),xlim=range(newdata))
lines(newdata, p$mean, lwd=1,col="red")
lines(newdata, p$lower95, lwd=1,lty=2)
lines(newdata, p$upper95, lwd=1,lty=2)

###############################################
### FIG 9a filtration
n <- 200
x <- seq(from=0, to=1, length=n)

covtype <- "gauss"
coef.cov <- c(theta <- 0.2)
sigma <- 1.5
trend <- c(intercept <- 0)
nugget <- 0.00000001   
formula <- ~1

model <- km(formula, design=data.frame(x=x), response=rep(0,n), covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2, nugget=nugget)
y <- simulate(model, nsim=200, newdata=NULL)

plot(x, y[1,], type="l", col=1, ylab="y", ylim=range(y))
for (i in 2:200) {
	lines(x, y[i,], col=i)
}

###############################################
### FIG 9b filtration
n <- 5
m <- 101
design <- seq(from=0, to=1, length=n)
response <- c(0.5, 0, 1.5, 3, 2)

covtype <- "gauss"
coef.cov <- 0.2
sigma <- 1.5

trend <- c(intercept <- 0)
model <- km(formula=~1, design=data.frame(x=design), response=response, 
            covtype=covtype, coef.trend=trend, coef.cov=coef.cov, 
            coef.var=sigma^2)

newdata <- seq(from=0, to=1, length=m)
nsim <- 40
y <- simulate(model, nsim=nsim, newdata=newdata, cond=TRUE, nugget.sim=1e-5)

plot(design, response, pch=19, cex=0.5, col="black",xlim=c(0,1),ylim=c(-1,4))
for (i in 1:nsim) {
	lines(newdata, y[i,], col=i)
}

###############################################
### FIG 10 combinaison lineaire de fonctions de base
n <- 5
m <- 101
design <- seq(from=0, to=1, length=n)
response <- c(0.5, 0, 1.5, 3, 2)
newdata <- seq(from=0, to=1, length=m)

covtype <- "gauss"
coef.cov <- 0.1
sigma <- 1.5

kk <- matrix(0,m,n)
for(i in 1:n){
kk[,i] <- dnorm(newdata-design[i], mean = 0, sd = coef.cov , log = FALSE)/3
}

plot(design, response, pch=19, cex=1, col="red",ylim=range(y),xlim=range(newdata))
lines(newdata,kk[,1],lty=1,lwd=2)
lines(newdata,kk[,2],lty=2,lwd=2)
lines(newdata,kk[,3],lty=3,lwd=2)
lines(newdata,kk[,4],lty=4,lwd=2)
lines(newdata,kk[,5],lty=5,lwd=2)


###############################################
### FIG 11 Influence du noyau sur les mod?les
n <- 5
m <- 101
design <- seq(from=0, to=1, length=n)
response <- c(0.5, 0, 1.5, 3, 2)
trend <- c(intercept <- 0)
newdata <- seq(from=0, to=1, length=m)

## 11a
covtype <- "gauss"
coef.cov <- 0.2
sigma <- 1.5

model <- km(formula=~1, design=data.frame(x=design), response=response, covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2)
y <- simulate(model, nsim=20, newdata=newdata, cond=TRUE, nugget.sim=1e-5)
p <- predict(model, newdata=newdata, type="SK")

plot(design, response, pch=19, cex=0.5, col="black",ylim=range(y),xlim=range(newdata))
lines(newdata, p$mean, lwd=1.5,col="red")
lines(newdata, p$lower95, lwd=1.5,lty=2)
lines(newdata, p$upper95, lwd=1.5,lty=2)
title("Noyau gaussien")

## 11b
covtype <- "exp"
coef.cov <- 0.2
sigma <- 1.5

model <- km(formula=~1, design=data.frame(x=design), response=response, covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2)
y <- simulate(model, nsim=20, newdata=newdata, cond=TRUE, nugget.sim=1e-5)
p <- predict(model, newdata=newdata, type="SK")

plot(design, response, pch=19, cex=0.5, col="black",ylim=range(y),xlim=range(newdata))
lines(newdata, p$mean, lwd=1.5,col="red")
lines(newdata, p$lower95, lwd=1.5,lty=2)
lines(newdata, p$upper95, lwd=1.5,lty=2)
title("Noyau exponentiel")


###################################################"
## Cas multidim
library(rgl) # librairie pour les graphiques 3D

### FIG 12

d <- 2; n <- 16
fact.design <- expand.grid(seq(0,1,length=3), seq(0,1,length=3))
fact.design <- data.frame(fact.design); names(fact.design)<-c("x1", "x2")
branin.resp <- data.frame(branin(fact.design)); names(branin.resp) <- "y" 

m1 <- km(~1, design=fact.design, response=branin.resp, covtype="gauss")

# predicting at testdata points
testdata <- expand.grid(s <- seq(0,1, length=15), s)
names(testdata)<-c("x1", "x2")
predicted.values.model1 <- predict(m1, testdata, "UK")

## 12a
open3d()
persp3d(s,s,matrix(branin(testdata)$x1,15,15),xlim=c(0,1),ylim=c(0,1),zlim=c(0,300),col="green",xlab="x",ylab="y",zlab="f")

## 12b
open3d()
plot3d(fact.design$x1,fact.design$x2, branin.resp$y,xlim=c(0,1),ylim=c(0,1),zlim=c(-200,400),cex=3,xlab="x",ylab="y",zlab="m et v")
surface3d(s,s,matrix(predicted.values.model1$mean,15,15),alpha=1,col="red")
surface3d(s,s,matrix(predicted.values.model1$upper95,15,15),alpha=0.5,col="lightblue")
surface3d(s,s,matrix(predicted.values.model1$lower95,15,15),alpha=0.5,col="lightblue")

## 12c
open3d()
persp3d(s,s,matrix(branin(testdata)$x1,15,15),xlim=c(0,1),ylim=c(0,1),zlim=c(-50,300),col="green",xlab="x",ylab="y",zlab="f et m")
surface3d(s,s,matrix(predicted.values.model1$mean,15,15),alpha=1,col="red")
points3d(fact.design$x1,fact.design$x2, branin.resp$y)

###############################################
### FIG 13 Influence des port?es
n <- 200
x <- seq(from=0, to=1, length=n)
trend <- c(intercept <- 0)
nugget <- 0.00000001   
formula <- ~1
covtype <- "gauss"

par(mfrow=c(1,3))

coef.cov <- c(theta <- 0.05)
sigma <- 1.5
model <- km(formula, design=data.frame(x=x), response=rep(0,n), covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2, nugget=nugget)
y <- simulate(model, nsim=10, newdata=NULL)
plot(x, y[1,], type="l", col=1, ylab="y", ylim=range(y))
title("theta = 0.05")
for (i in 2:5) {
	lines(x, y[i,], col=i)
}

coef.cov <- c(theta <- 0.2)
sigma <- 1.5
model <- km(formula, design=data.frame(x=x), response=rep(0,n), covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2, nugget=nugget)
y <- simulate(model, nsim=10, newdata=NULL)
plot(x, y[1,], type="l", col=1, ylab="y", ylim=range(y))
title("theta = 0.2")
for (i in 2:5) {
	lines(x, y[i,], col=i)
}

coef.cov <- c(theta <- 0.6)
sigma <- 1.5
model <- km(formula, design=data.frame(x=x), response=rep(0,n), covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2, nugget=nugget)
y <- simulate(model, nsim=10, newdata=NULL)
plot(x, y[1,], type="l", col=1, ylab="y", ylim=range(y))
title("theta = 0.6")
for (i in 2:8) {
	lines(x, y[i,], col=i)
}

par(mfrow=c(1,1))

###############################################
### FIG 14 Influence de la variance
n <- 200
x <- seq(from=0, to=1, length=n)
trend <- c(intercept <- 0)
nugget <- 0.00000001   
formula <- ~1
covtype <- "gauss"

par(mfrow=c(1,3))

coef.cov <- c(theta <- 0.2)
sigma <- 0.5
model <- km(formula, design=data.frame(x=x), response=rep(0,n), covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2, nugget=nugget)
y <- simulate(model, nsim=10, newdata=NULL)
plot(x, y[1,], type="l", col=1, ylab="y", ylim=c(-10,10))
title("sigma = 0.5")
for (i in 2:5) {
	lines(x, y[i,], col=i)
}

coef.cov <- c(theta <- 0.2)
sigma <- 2
model <- km(formula, design=data.frame(x=x), response=rep(0,n), covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2, nugget=nugget)
y <- simulate(model, nsim=10, newdata=NULL)
plot(x, y[1,], type="l", col=1, ylab="y", ylim=c(-10,10))
title("sigma = 2")
for (i in 2:5) {
	lines(x, y[i,], col=i)
}

coef.cov <- c(theta <- 0.2)
sigma <- 4
model <- km(formula, design=data.frame(x=x), response=rep(0,n), covtype=covtype, coef.trend=trend, coef.cov=coef.cov, coef.var=sigma^2, nugget=nugget)
y <- simulate(model, nsim=10, newdata=NULL)
plot(x, y[1,], type="l", col=1, ylab="y", ylim=c(-10,10))
title("sigma =  4")
for (i in 2:8) {
	lines(x, y[i,], col=i)
}

par(mfrow=c(1,1))



