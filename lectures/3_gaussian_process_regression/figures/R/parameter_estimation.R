library(tikzDevice)
library(DiceKriging)

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

kMat52 <- function(x,y,param=c(1,.2)){
  d <- sqrt(5)*abs(outer(x,y,"-"))/param[2]
  return(param[1]*(1 + d +1/3*d^2)*exp(-d))
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
N <- c(5,10,15,20)

SIG2 <- THET <- matrix(0,2,4)
for(j in 1:4){
  n <- N[j]
  X <- seq(from=1/2/n, to=1-1/2/n, length=n)
  sig2 <- thet <- rep(0,100)
  for(i in 1:100){
    F <- mvrnorm(1,rep(0,n),kMat52(X,X))
    model <- km(formula<- ~1, design=data.frame(x=X), response=data.frame(y=F), covtype="matern5_2",coef.trend=0)
    sig2[i] <- model@covariance@sd2
    thet[i] <- model@covariance@range.val
  }
  SIG2[,j] <- c(mean(sig2),sd(sig2))
  THET[,j] <- c(mean(thet),sd(thet))
}

round(SIG2,2)
round(THET,2)



