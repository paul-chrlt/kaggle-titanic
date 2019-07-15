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

library(stringr)

featurecrator <- function(dataset){
    ### child under 20
    dataset$isChild <- dataset$Age < 20
    ### has family
    dataset$hasFamily <- dataset$Parch != 0
    ### has a partner (sibling or spouse)
    dataset$hasPartner <- dataset$SibSp > 0
    ### name length
    dataset$nameLength <- str_length(dataset$Name)
    dataset
}

trainset <- featurecrator(trainset)
testset <- featurecrator(testset)


### dataset preparation

outcome <- trainset$Survived
modeltrainset <- trainset[,c(2,3,5,6,7,8,10,12,13,14,15,16)]

traincontrol <- trainControl()
## training

### GLM, 82% accuracy

glmmodel <- glm(Survived~.,data = modeltrainset,family = "binomial")
glmprediction <- predict(glmmodel,testset)
glmprediction[glmprediction>.5] <- 1
glmprediction[glmprediction<.5] <- 0
confusionMatrix(as.factor(glmprediction),as.factor(testset$Survived))

### adaBoost, 82% accuracy
adaboostModel <- train(modeltrainset[,-1],
                    outcome,
                    method = "adaboost")
adaboostprediction <- predict(adaboostModel,testset)
confusionMatrix(adaboostprediction,testset$Survived)

### Random Forest, 85% accuracy

rfModel <- train(modeltrainset[,-1],
                 outcome,
                 method = "rf")
rfPrediction <- predict(rfModel,testset)
confusionMatrix(rfPrediction,testset$Survived)

## Vote, 84% accuracy

predictionsVote <- data.frame(glmprediction,adaboostprediction,rfPrediction)
predictionsVote$adaboostprediction <- as.integer(as.character(predictionsVote$adaboostprediction))
predictionsVote$rfPrediction <- as.integer(as.character(predictionsVote$rfPrediction))
predictionsVote$vote <- as.integer((predictionsVote$glmprediction+predictionsVote$adaboostprediction+predictionsVote$rfPrediction)>1)

confusionMatrix(as.factor(predictionsVote$vote),testset$Survived)
