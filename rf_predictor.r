
diabetes <- read.csv('all_data_important_columns.csv', header=TRUE)
# Drop "state" as a variable
diabetes <- diabetes[names(diabetes) != 'State']
# Convert Gender to indicator for F
diabetes$Gender <- as.integer(diabetes$Gender == 'F')
diabetes$DMIndicator <- factor(diabetes$DMIndicator)
shuffled <- sample(nrow(diabetes))
train_index <- shuffled[1:8000]
test_index <- shuffled[-(1:8000)]
diabetes_train <- diabetes[train_index, ]
diabetes_test <- diabetes[test_index, ]
library(randomForest)
set.seed(16)
diabetes_rf <- randomForest(x=diabetes_train[-1], 
                            y=diabetes_train$DMIndicator,
                            sampsize=c(1000, 1000), replace=FALSE)
diabetes_rf