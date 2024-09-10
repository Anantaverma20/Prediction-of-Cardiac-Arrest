## Importing Library
install.packages("xgboost")
library(xgboost)
library(modeldata)
library(dplyr)
library(caret)
library(corrplot)

## Importing Data
data <- read.csv("E:\\DS project\\all code and algorithms\\final_data.csv",sep = ",",header = T, na.strings = "?")


## data Manipulation

x <- data %>% select(Depression, HTN,	Smoking,	F_History,	Diabetes,	BP,	hemoglobin,	platelete_count,	cholestrol,	Diagnosis) 

unlist(x)

daf <- cor(x)

corrplot(daf, order = "AOE", method = "color", addCoef.col = "black")

summary(x)

is.null(x)

unique.data.frame(x)


## data splitting

set.seed(1234)

ind <- sample(2, nrow(x), replace = T, prob = c(0.7, 0.3))
train <- x[ind==1,]
test <- x[ind==2,]


## matrix creation
train_x = data.matrix(train[, -1])   
train_y = train[,1]

test_x = data.matrix(test[, -1])
test_y = test[, 1]

## final matrix
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#define watchlist
watchlist = list(train=xgb_train, test=xgb_test)


## setting parameter
param <-  list(set.seed = 1500, 
               eval_mertic = "mlogloss", 
               objective = "binary:logistic")

## XGboost
model <- xgb.train(data = xgb_train,
                   params = param,
                   nrounds = 100,
                   watchlist = watchlist,
                   eta = 0.01)

timing = system.time({
## testing model
xgb.plot.shap(data = train_x,
              model = model,
              top_n =5)
})
print(timing)
e <- data.frame(model$evaluation_log)
plot(e$iter, e$train_logloss,  col = 'blue')
lines(e$iter, e$test_logloss, col = "red")




pred_y = predict(model, xgb_test)

mean((test_y - pred_y)^2) #mse
caret::MAE(test_y, pred_y) #mae
caret::RMSE(test_y, pred_y) #rmse


## importance
imp <- xgb.importance(colnames(xgb_train),
                      model = model)

xgb.plot.importance(imp)


## confusion matrix

xgbpred <- predict(model,xgb_test)
xgbpred <- ifelse(xgbpred > 0.5,1,0)


confusionMatrix (as.factor(xgbpred), as.factor(test_y))
library(Matrix)

time <- system.time({

# Number of splits for stratified k-fold cross-validation
n_splits <- 10

# Initialize an empty vector to store cross-validation results
cv_results <- numeric(n_splits)

# Create indices for stratified k-fold cross-validation
set.seed(123)  # Set a random seed for reproducibility
folds <- createFolds(train_y, k = n_splits, list = TRUE)

# Perform stratified k-fold cross-validation
for (fold in 1:n_splits) {
  # Split the data into training and validation sets
  train_indices <- unlist(folds[-fold])
  val_indices <- unlist(folds[fold])
  
  X_train_fold <- train_x[train_indices, ]
  y_train_fold <- train_y[train_indices]
  X_val_fold <- train_x[val_indices, ]
  y_val_fold <- train_y[val_indices]
  
  # Convert data to xgb.DMatrix format
  dtrain <- xgb.DMatrix(data = as.matrix(X_train_fold), label = y_train_fold)
  dval <- xgb.DMatrix(data = as.matrix(X_val_fold), label = y_val_fold)
  
  # Specify XGBoost parameters
  params <- list(
    objective = "binary:logistic",  # for binary classification
    eval_metric = "logloss",       # evaluation metric
    max_depth = 3,                 # maximum tree depth
    eta = 0.1                      # learning rate
  )
  
  # Train an XGBoost model
  model <- xgboost(params = params, data = dtrain, nrounds = 100, verbose = 0)
  
  # Evaluate the model on the validation set
  y_pred <- predict(model, dval)
  
  # Calculate the accuracy or other relevant metric
  accuracy <- sum(round(y_pred) == y_val_fold) / length(y_val_fold)
  
  # Store the cross-validation result
  cv_results[fold] <- accuracy
}

# Calculate and print the mean cross-validation accuracy
mean_accuracy <- mean(cv_results)
cat("Mean Cross-Validation Accuracy:", mean_accuracy, "\n")
})


print(time)

# Create a bar plot of cross-validation results
barplot(cv_results, names.arg = 1:n_splits, xlab = "Fold", ylab = "Accuracy", 
        main = "Cross-Validation Accuracy", ylim = c(0, 1))
