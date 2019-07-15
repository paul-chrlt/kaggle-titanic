## libraries

library(caret)

## data import

trainSource <-read.csv("./datasource/train.csv",stringsAsFactors = FALSE) 
testSource <- read.csv("./datasource/test.csv",stringsAsFactors = FALSE)

## Splitting dataset

trainindex <- createDataPartition(trainSource$Survived,p=0.8,list = FALSE)
trainset <- trainSource[trainindex,]
testset <- trainSource[-trainindex,]

## data cleaning

meanage <- mean(trainSource$Age,na.rm = TRUE)

formatter <- function(dataset){
    dataset$Survived <- as.factor(dataset$Survived)
    dataset[is.na(dataset$Age),]$Age <- meanage
    dataset$Pclass <- as.factor(dataset$Pclass)
    dataset$Sex <- as.factor(dataset$Sex)
    dataset$Embarked <- as.factor(dataset$Embarked)
    dataset
}

trainset <- formatter(trainset)
testset <- formatter(testset)

## feature engineering



### dataset preparation

outcome <- trainset$Survived
modeltrainset <- trainset[,c(3,5,6,7,8,10,12)]

traincontrol <- trainControl()

## training

### GLM, 81% accuracy

glmmodel <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data = trainset,family = "binomial")
glmprediction <- predict(glmmodel,testset)
glmprediction[glmprediction>.5] <- 1
glmprediction[glmprediction<.5] <- 0
confusionMatrix(as.factor(glmprediction),as.factor(testset$Survived))

### adaBoost, 82% accuracy
adaboostModel <- train(modeltrainset,
                    outcome,
                    method = "adaboost")
adaboostprediction <- predict(adaboostModel,testset)
confusionMatrix(adaboostprediction,testset$Survived)

### Random Forest, 85% accuracy

rfModel <- train(modeltrainset,
                 outcome,
                 method = "rf")
rfPrediction <- predict(rfModel,testset)
confusionMatrix(rfPrediction,testset$Survived)
