library(tikzDevice)
library(MASS)

lightblue <- rgb(114/255,159/255,207/255,.3)
darkblue <- rgb(32/255,74/255,135/255,1)
darkbluetr <- rgb(32/255,74/255,135/255,.3)

darkPurple <- "#5c3566"
darkBlue <- "#204a87"
darkGreen <- "#4e9a06"
darkChocolate <- "#8f5902"
darkRed  <- "#a40000"
darkOrange <- "#ce5c00"
darkButter <- "#c4a000"

plotGPR <- function(x,GP,ylab="$Z(x)$"){
  m <- GP[[1]]
  low95 <- GP[[3]]
  upp95 <- GP[[4]]
  par(mar=c(4.5,5.1,1.5,1.5))
  plot(x, m, type="n", xlab="$x$",ylab=ylab, ylim=range(low95-0.5,upp95+0.5),cex.axis=1.5,cex.lab=2)
  polygon(c(x,rev(x)),c(upp95,rev(low95)),border=NA,col=lightblue)
  lines(x,m,col=darkblue,lwd=3)
  lines(x,low95,col=darkbluetr)  
  lines(x,upp95,col=darkbluetr)  
}

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

kWhite <- function(x,y,param=1){
  return(param*outer(x,y,"-")==0)
}

GPR <- function(x,X,F,kern,param,kernNoise=kWhite,paramNoise=0){
  m <- kern(x,X,param)%*%solve(kern(X,X,param))%*%F
  K <- kern(x,x,param) - kern(x,X,param)%*%solve(kern(X,X,param))%*%kern(X,x,param)  
  upp95 <- m + 1.96*sqrt(pmax(0,diag(K)))
  low95 <- m - 1.96*sqrt(pmax(0,diag(K)))
  return(list(m,K,low95,upp95))
}

crossValid <- function(x,X,F,kern,param,kernNoise=kWhite,paramNoise=0){
  pred <- varpred <- rep(0,length(F))
  for(i in 1:length(F)){
    GP <- GPR(X[i],X[-i],F[-i],kern,param)
    pred[i] <- GP[[1]]
    varpred[i] <- GP[[2]]
  }
  return(list(pred,varpred))
}

Q2 <- function(obs,pred){
  return(1-sum((obs-pred)^2)/sum((obs-mean(obs))^2))
}

##################################################################"
##################################################################"
### test set
n <- 10
nt <- 20
x <- seq(from=0, to=1, length=101)
indX <- seq(from=5, to=95, length=n)
indXt <- round(runif(nt)*101-0.5)

X <- x[indX]
Xt <- x[indXt]
Ftot <- mvrnorm(1,rep(0,n+nt),kMat32(c(X,Xt),c(X,Xt)))
F <- Ftot[1:n]
Ft <- Ftot[(n+1):(n+nt)]

GP <- GPR(x,X,F,kMat32,c(1,.2))

tikz('VALID_testset.tex', standAlone = TRUE, width=8, height=5)
plotGPR(x,GP,"$Z(x)|Z(X)=F$")
points(X, F, pch=4, cex=1,lwd=3)
points(Xt, Ft, pch=4, cex=1,lwd=3,col='red')
dev.off()
tools::texi2dvi('VALID_testset.tex',pdf=T)

## compute RSS and Q2
pred <- GPR(Xt,X,F,kMat32,c(1,.2))[[1]]
MSE <-mean((pred - Ft)^2)
Q2(Ft,pred)

## standardise errors
n <- 10
nt <- 101
x <- seq(from=0, to=1, length=101)
indX <- seq(from=5, to=95, length=n)
indXt <- 1:101

X <- x[indX]
Xt <- x[indXt]
Ftot <- mvrnorm(1,rep(0,n+nt),kMat32(c(X,Xt),c(X,Xt)))
F <- Ftot[1:n]
Ft <- Ftot[(n+1):(n+nt)]

pred <- GPR(Xt,X,F,kMat32,c(1,.2))[[1]]
predvar <- GPR(Xt,X,F,kMat32,c(1,.2))[[2]]
Yiid <- chol(solve(predvar+diag(rep(1e-10,nt)))) %*% (Ft-pred)

par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
tikz('VALID_hist.tex', standAlone = TRUE, width=5, height=5)
hist(Yiid,20,freq=FALSE,main='',xlab='standardised residuals')
lines(seq(-4,4,length=100),dnorm(seq(-4,4,length=100)))
dev.off()
tools::texi2dvi('VALID_hist.tex',pdf=T)

tikz('VALID_qqplot.tex', standAlone = TRUE, width=5, height=5)
qqnorm(Yiid)
lines(range(Yiid),range(Yiid))
dev.off()
tools::texi2dvi('VALID_qqplot.tex',pdf=T)


## cross validation
GP <- GPR(x,X,F,kGauss,c(1,.1))
pred <- crossValid(x,X,F,kMat32,c(1,.2))[[1]]
mean((pred - F)^2)
Q2(F,pred)

tikz('VALID_crossval0.tex', standAlone = TRUE, width=8, height=5)
GP <- GPR(x,X,F,kMat32,c(1,.2))
plotGPR(x,GP,"$Z(x)|Z(X)=F$")
points(X,F, pch=4, cex=1,lwd=3)
dev.off()
tools::texi2dvi('VALID_crossval0.tex',pdf=T)

tikz('VALID_crossval1.tex', standAlone = TRUE, width=8, height=5)
ind <- 1
GP <- GPR(x,X[-ind],F[-ind],kMat32,c(1,.2))
#GP <- GPR(x,X,F,kGauss,c(1,.1))
plotGPR(x,GP,"$Z(x)|Z(X)=F$")
points(X[-ind],F[-ind], pch=4, cex=1,lwd=3)
points(X[ind],F[ind], pch=4, cex=1,lwd=3,col='red')
dev.off()
tools::texi2dvi('VALID_crossval1.tex',pdf=T)

## standardised residuals
pred <- crossValid(x,X,F,kMat32,c(1,.2))[[1]]
varpred <- crossValid(x,X,F,kMat32,c(1,.2))[[2]]
Yiid <- (F-pred)/sqrt(varpred)

par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.5,cex.lab=2)
tikz('VALID_crossvalhist.tex', standAlone = TRUE, width=5, height=5)
hist(Yiid,freq=FALSE,main='',xlab='standardised residuals',ylim=c(0,.45))
lines(seq(-4,4,length=100),dnorm(seq(-4,4,length=100)))
dev.off()
tools::texi2dvi('VALID_crossvalhist.tex',pdf=T)

tikz('VALID_crossvalqqplot.tex', standAlone = TRUE, width=5, height=5)
qqnorm(Yiid)
lines(range(Yiid),range(Yiid))
dev.off()
tools::texi2dvi('VALID_crossvalqqplot.tex',pdf=T)





