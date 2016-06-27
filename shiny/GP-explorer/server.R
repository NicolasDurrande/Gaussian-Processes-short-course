library(shiny)

# Define server logic required to plot various variables against mpg
shinyServer(function(input, output) {
  
  XY <- reactiveValues(
    X = c(.1,.3,.5,.7,.9),
    Y = sin(4*pi*c(.1,.3,.5,.7,.9)) + 2*c(.1,.3,.5,.7,.9)
  )
  
  observeEvent(input$plotSample_click, {
    if(input$addPts){
    XY$X <- c(XY$X,input$plotSample_click$x)
    XY$Y <- c(XY$Y,input$plotSample_click$y)
    }else{
      nearPt <- nearPoints(XY$X,XY$Y,input$plotSample_click$x,input$plotSample_click$y)
      if(length(nearPt)){
        XY$X <- XY$X[-nearPt]
        XY$Y <- XY$Y[-nearPt]
      }
    }
  })
  
  output$mean <- renderPlot({
    source('functions.R')
    d <- c(0,1)        # domaine
    n_grid <- input$n_grid
    x <- seq(d[1],d[2],length.out=n_grid) # prediction grid
    
    b0 <- b1 <- b2 <- 0
    if(input$mean>0) b0 <- input$b0
    if(input$mean>1) b1 <- input$b1
    if(input$mean>2) b2 <- input$b2
    
    mean <- b0 + b1*x + b2*x^2
    
    par_dflt <- par(no.readonly=TRUE)
    par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.2,cex.lab=1.5)
    plot(x,mean,
         type='l',col=darkblue,lwd=2,
         xlab='x',ylab=paste0("mu(x)"),main="")
    par(par_dflt)
  })

  output$kernel <- renderPlot({
    source('functions.R')
    d <- c(0,1)        # domaine
    n_grid <- input$n_grid

    kernel <- get_kernel(input$kernel)
    param <- c(input$sig2,input$theta)
    
    x <- matrix(seq(d[1],d[2],length.out=n_grid)) # prediction grid
    xnew <- matrix(mean(d)) 
    
    par_dflt <- par(no.readonly=TRUE)
    par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.2,cex.lab=1.5)
    plot(x,kernel(x,xnew,param),
         type='l',col=darkblue,lwd=2,
         ylim=c(0,input$sig2*1.1),
         xlab='x',ylab=paste0("k(x,",xnew,")"),main="")
    lines(rep(xnew,2),c(0,input$sig2*1.1),lty=2,col="gray")
    par(par_dflt)
  })
  
  output$priorSamples <- renderPlot({
    source('functions.R')
    d <- c(0,1)        # domaine
    n_grid <- input$n_grid
    n_sample <- input$n_sample
    x <- matrix(seq(d[1],d[2],length.out=n_grid)) # prediction grid
    
    b0 <- b1 <- b2 <- 0
    if(input$mean>0) b0 <- input$b0
    if(input$mean>1) b1 <- input$b1
    if(input$mean>2) b2 <- input$b2
    
    mean <- b0 + b1*x + b2*x^2
    
    kernel <- get_kernel(input$kernel)
    param <- c(input$sig2,input$theta)
    K <- kernel(x,x,param)
    
    Z <- t(mvrnorm(n_sample,c(mean),K))

    par_dflt <- par(no.readonly=TRUE)
    par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.2,cex.lab=1.5)
    matplot(x,Z,type='l',lty=1,xlab="x",ylab="Z(x)",main="",col=brewer.pal(8,"RdBu"))
    par(par_dflt)
  })
  
  output$posteriorSamples <- renderPlot({
    source('functions.R')
    d <- c(0,1)        # domaine
    n_grid <- input$n_grid
    n_sample <- input$n_sample
    x <- matrix(seq(d[1],d[2],length.out=n_grid)) # prediction grid
    
    X <- matrix(XY$X)
    Y <- matrix(XY$Y)
    
    b0 <- b1 <- b2 <- 0
    if(input$mean>0) b0 <- input$b0
    if(input$mean>1) b1 <- input$b1
    if(input$mean>2) b2 <- input$b2
    
    mean <- b0 + b1*x + b2*x^2
    Ymean <- b0 + b1*X + b2*X^2
    
    kernel <- get_kernel(input$kernel)
    param <- c(input$sig2,input$theta)
    
    Kxx <- kernel(x,x,param)
    KxX <- kernel(x,X,param)
    KXX_1 <- solve(kernel(X,X,param))
    
    cond_mean <- mean+KxX %*% KXX_1 %*% (Y-Ymean)
    cond_var <- Kxx - KxX %*% KXX_1 %*% t(KxX)
    
    Z <- t(mvrnorm(n_sample,c(cond_mean),cond_var))
    
    par(mar=c(4.5,5.1,1.5,1.5),cex.axis=1.2,cex.lab=1.5)
    plot(x[,1],Z[,1],type='l',lty=1,xlab="x",ylab="Z(x)|Z(X)=Y",main="",col=brewer.pal(8,"RdBu"))
    matplot(x,Z,type='l',lty=1,xlab="x",ylab="Z(x)|Z(X)=Y",main="",col=brewer.pal(8,"RdBu"))
    points(X,Y,pch=4, cex=1,lwd=3)
  })
  
  output$text1 <- renderText({ 
    c(input$b0,input$b1,input$b2)
  })
  
  output$priormean <- renderUI({
    if(input$mean==0) return(withMathJax("$$\\mu(x)=0 $$"))
    if(input$mean==1) return(withMathJax("$$\\mu(x)=\\beta_0$$"))
    if(input$mean==2) return(withMathJax("$$\\mu(x)=\\beta_0 + \\beta_1 x$$"))
    if(input$mean==3) return(withMathJax("$$\\mu(x)=\\beta_0  + \\beta_1 x  + \\beta_2 x^2$$"))
  })
  
  output$priorkernel <- renderUI({
    if(input$kernel=="kExp") return(withMathJax("$$k(x,y)=\\sigma^2 \\exp \\left(-\\frac{\\left|x-y\\right|}{\\theta} \\right)$$"))
    if(input$kernel=="kMat32") return(withMathJax("$$k(x,y)=\\sigma^2 \\left(1 + \\sqrt{3} \\frac{\\left|x-y\\right|}{\\theta} \\right) \\exp \\left(- \\frac{\\left|x-y\\right|}{\\theta} \\right)$$"))
    if(input$kernel=="kMat52") return(withMathJax("$$k(x,y)=\\sigma^2 \\left(1 + \\sqrt{5} \\frac{\\left|x-y\\right|}{\\theta} + \\frac{5}{3} \\frac{\\left|x-y\\right|^2}{\\theta^2} \\right) \\exp \\left(- \\sqrt{5} \\frac{\\left|x-y\\right|}{\\theta} \\right)$$"))
    if(input$kernel=="kGauss") return(withMathJax("$$k(x,y)=\\sigma^2 \\exp \\left(-\\frac{\\left(x-y\\right)^2}{2 \\, \\theta^2} \\right)$$"))
    if(input$kernel=="kBrown") return(withMathJax("$$k(x,y)=\\sigma^2 \\min(x,y))$$"))
  })

#   output$info <- renderText({
#     paste0("x=", XY$X, "\ny=", XY$Y)
#   })
  
})