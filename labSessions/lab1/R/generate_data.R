library(rgl)
library(shiny)
library(DiceDesign)
runGitHub("shinyApps",username="NicolasDurrande",subdir="catapult")

n <- 30
X <- lhsDesign(n,4)$design
colnames(X) <- paste0("x",1:4)

Y <- rep(0,n)
for(i in 1:n){
  Y[i] <- runExperiment(X[i,],1)[1]
}

data <- round(cbind(X,Y),2)
pairs(data)
plot3d(X[,1],X[,2],Y)

write.table(data, "lab1_data.csv", sep=", ", row.names=FALSE, col.names=FALSE)

reg <- lm(formula = Y ~ x1 + x2 + x3 + x4 + I(x1^2) + I(x2^2) + I(x3^2) + I(x4^2), data = data.frame(data))
summary(reg)
