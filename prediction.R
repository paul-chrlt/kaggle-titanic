## libraries

library(caret)

## data import

trainSource <-read.csv("./datasource/train.csv",stringsAsFactors = FALSE) 
kaggletestSource <- read.csv("./datasource/test.csv",stringsAsFactors = FALSE)

## Splitting dataset

trainindex <- createDataPartition(trainSource$Survived,p=0.8,list = FALSE)
trainset <- trainSource[trainindex,]
testset <- trainSource[-trainindex,]

## data cleaning

meanage <- mean(trainSource$Age,na.rm = TRUE)
meanfare <- mean(trainSource$Fare,na.rm=TRUE)

formatter <- function(dataset){
    if(!is.null(dataset$Survived)){
        dataset$Survived <- as.factor(dataset$Survived)
    }
    if(length(dataset[is.na(dataset$Age),]$Age)>0){
        dataset[is.na(dataset$Age),]$Age <- meanage
    }
    if(length(dataset[is.na(dataset$Fare),]$Fare)>0){
        dataset[is.na(dataset$Fare),]$Fare <- meanfare
    }
    dataset$Pclass <- as.factor(dataset$Pclass)
    dataset$Sex <- as.factor(dataset$Sex)
    dataset$Embarked <- as.factor(dataset$Embarked)
    dataset
}

trainset <- formatter(trainset)
testset <- formatter(testset)
kaggletest <- formatter(kaggletestSource)

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
    ### has a cabin
    dataset$hasCabin <- dataset$Cabin != ""
    dataset
}

trainset <- featurecrator(trainset)
testset <- featurecrator(testset)
kaggletest <- featurecrator(kaggletest)

### dataset preparation

modeltrainset <- trainset[,c(2,3,5,6,7,8,10,12,13,14,15,16)]

traincontrol <- trainControl()
## training

### GLM, 82% accuracy

glmmodel <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+isChild+hasFamily+hasPartner+nameLength+hasCabin,data = trainset,family = "binomial")
glmprediction <- predict(glmmodel,testset)
glmprediction[glmprediction>.5] <- 1
glmprediction[glmprediction<.5] <- 0
confusionMatrix(as.factor(glmprediction),as.factor(testset$Survived))

### adaBoost, 82% accuracy
adaboostModel <- train(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+isChild+hasFamily+hasPartner+nameLength+hasCabin,
                    method = "adaboost",
                    data=trainset)
adaboostprediction <- predict(adaboostModel,testset)
confusionMatrix(adaboostprediction,testset$Survived)

### Random Forest, 85% accuracy

rfModel <- train(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+isChild+hasFamily+hasPartner+nameLength+hasCabin,
                 method="rf",
                 data = trainset)

rfPrediction <- predict(rfModel,testset)
confusionMatrix(rfPrediction,testset$Survived)

## Vote, 84% accuracy

predictionsVote <- data.frame(glmprediction,adaboostprediction,rfPrediction)
predictionsVote$adaboostprediction <- as.integer(as.character(predictionsVote$adaboostprediction))
predictionsVote$rfPrediction <- as.integer(as.character(predictionsVote$rfPrediction))
predictionsVote$vote <- as.integer((predictionsVote$glmprediction+predictionsVote$adaboostprediction+predictionsVote$rfPrediction)>1)

confusionMatrix(as.factor(predictionsVote$vote),testset$Survived)

## Predict Kaggle data
kglmprediction <- predict(glmmodel,kaggletest)
kglmprediction[kglmprediction>.5] <- 1
kglmprediction[kglmprediction<.5] <- 0

kadaboostprediction <- predict(adaboostModel,kaggletest)

krfPrediction <- predict(rfModel,kaggletest)

kpredictionsVote <- data.frame(kglmprediction,kadaboostprediction,krfPrediction)
kpredictionsVote$kadaboostprediction <- as.integer(as.character(kpredictionsVote$kadaboostprediction))
kpredictionsVote$krfPrediction <- as.integer(as.character(kpredictionsVote$krfPrediction))
kpredictionsVote$vote <- as.integer((kpredictionsVote$kglmprediction+kpredictionsVote$kadaboostprediction+kpredictionsVote$krfPrediction)>1)

ksubmission <- cbind(kaggletest$PassengerId,kpredictionsVote$vote)
write.csv(ksubmission,file="submission.csv",row.names = FALSE)
