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

## training

### control

control <- trainControl(method="repeatedcv", number = 30, repeats = 5,search = "grid")

### GLM, 82% accuracy

glmModel <-
    train(
        Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + isChild +
            hasFamily + hasPartner + nameLength + hasCabin,
        method = "glm",
        family = "binomial",
        data = trainset,
        trControl = control
    )
glmprediction <- predict(glmModel,testset)
confusionMatrix(glmprediction,testset$Survived)

### adaBoost, 82% accuracy
adaboostModel <- train(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+isChild+hasFamily+hasPartner+nameLength+hasCabin,
                    method = "adaboost",
                    data=trainset,
                    trControl=control)
adaboostprediction <- predict(adaboostModel,testset)
confusionMatrix(adaboostprediction,testset$Survived)

### Random Forest, 85% accuracy

rfModel <- train(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+isChild+hasFamily+hasPartner+nameLength+hasCabin,
                 method="rf",
                 data = trainset,
                 trControl=control)

rfPrediction <- predict(rfModel,testset)
confusionMatrix(rfPrediction,testset$Survived)

## Vote, 84% accuracy

predictionsVote <- data.frame(glmprediction,adaboostprediction,rfPrediction)
predictionsVote$glmprediction <- as.integer(as.character(predictionsVote$glmprediction))
predictionsVote$adaboostprediction <- as.integer(as.character(predictionsVote$adaboostprediction))
predictionsVote$rfPrediction <- as.integer(as.character(predictionsVote$rfPrediction))
predictionsVote$vote <- as.integer((predictionsVote$glmprediction+predictionsVote$adaboostprediction+predictionsVote$rfPrediction)>1)

confusionMatrix(as.factor(predictionsVote$vote),testset$Survived)

## Predict Kaggle data

kglmprediction <- predict(glmModel,kaggletest)

kadaboostprediction <- predict(adaboostModel,kaggletest)

krfPrediction <- predict(rfModel,kaggletest)

kpredictionsVote <- data.frame(kglmprediction,kadaboostprediction,krfPrediction)
kpredictionsVote$kglmprediction <- as.integer(as.character(kpredictionsVote$kglmprediction))
kpredictionsVote$kadaboostprediction <- as.integer(as.character(kpredictionsVote$kadaboostprediction))
kpredictionsVote$krfPrediction <- as.integer(as.character(kpredictionsVote$krfPrediction))
kpredictionsVote$vote <- as.integer((kpredictionsVote$kglmprediction+kpredictionsVote$kadaboostprediction+kpredictionsVote$krfPrediction)>1)

ksubmission <- data.frame(kaggletest$PassengerId,kpredictionsVote$vote)
names(ksubmission) <- c("PassengerId","Survived")
write.csv(ksubmission,file="submission.csv",row.names = FALSE)
