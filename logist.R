#Installing the package
install.packages("caTools")    # For Logistic regression
install.packages("ROCR")       # For ROC curve to evaluate model

# Loading package
library(caTools)
library(ROCR) 
data=read.csv("C:\\Users\\chand\\OneDrive\\Desktop\\moc data final.csv", header=T, stringsAsFactors=T)
# Splitting dataset
split <- sample.split(data, SplitRatio = 0.80)
split

train_reg <- subset(data, split == "TRUE")
test_reg <- subset(data, split == "FALSE")
attach(data)

# Training model
logistic_model <- glm(diagnosis ~Diabetes + cholestrol, 
                      data = train_reg, 
                      family = "binomial")

logistic_model


# Summary
summary(logistic_model)
options(warn=-1)      #turn off warnings


# Predict test data based on model
predict_reg <- predict(logistic_model, 
                       test_reg, type = "response")

predict_reg  

# Changing probabilities
predict_reg <- ifelse(predict_reg >0.5, 1, 0)

# Evaluating model accuracy
# using confusion matrix
table(test_reg$diagnosis, predict_reg)

missing_classerr <- mean(predict_reg != test_reg$diagnosis)
print(paste('Accuracy =', 1 - missing_classerr))
confusionMatrix=(predict_reg)
confusionMatrix
# ROC-AUC Curve
ROCPred <- prediction(predict_reg, test_reg$HTN) 
ROCPer <- performance(ROCPred, measure = "tpr", 
                      x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1.99]]
auc

# Plotting curve
plot(ROCPer)
plot(ROCPer, colorize = TRUE, 
     print.cutoffs.at = seq(1, by = 1), 
     main = "ROC CURVE")
abline(a = 0, b = 1)

auc <- round(auc, 7)
legend(.6, .4, auc, title = "AUC", cex = 1)

