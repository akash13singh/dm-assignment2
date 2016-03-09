dataset <- read.csv("~/TUE/Quartile3/DataMining/dm-assignment2/dataset.csv", sep = ",")

losses <- list()
for(i in 1:20) {
  sample <- sample.split(dataset$class, SplitRatio = .8)
  train <- subset(dataset, sample == T)
  test <- subset(dataset, sample == F)
  
  tree <- J48(class~., data = train, control = Weka_control(C=0.5))
  pred <- predict(tree, test)
  conf = table(pred, as.factor(test$class))
  loss <- 1 - sum(diag(conf))/nrow(test)
  losses[length(losses) + 1] <- loss
}

print(paste("mean loss= ", mean(as.numeric(losses))))
print(paste("variance loss= ", var(as.numeric(losses))))
