## data import

trainSource <-read.csv("./datasource/train.csv",stringsAsFactors = FALSE) 
testSource <- read.csv("./datasource/test.csv",stringsAsFactors = FALSE)

## Splitting dataset

trainindex <- createDataPartition(trainSource$Survived,p=0.8,list = FALSE)
trainset <- trainSource[trainindex,]
testset <- trainSource[-trainindex,]

## date cleaning

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

## training

model <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data = trainset)

prediction <- predict(model,testset)
prediction <- round(prediction)

confusionMatrix(as.factor(prediction),as.factor(testset$Survived))
## 81%
library(caret)
outcome <- trainset$Survived
modeltrainset <- trainset[,c(3,5,6,7,8,10,12)]
traincontrol <- trainControl()
adaboostModel <- train(modeltrainset,
                    outcome,
                    method = "adaboost")
adaboostprediction <- predict(adaboostModel,testset)
confusionMatrix(adaboostprediction,testset$Survived)
## 79%
rfModel <- train(modeltrainset,
                 outcome,
                 method = "rf")
rfPrediction <- predict(rfModel,testset)
## 82%
