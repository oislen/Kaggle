#########################
## Titanic Competition ##
#########################

########################
## load training data ##
########################

getwd()
setwd(dir = "C:/Users/Oisin/Documents/Kaggle/Titanic Competition")
data <- read.csv(file = "train.csv")
attach(data)
set.seed(seed = 1234)

summary(data)
str(data)
head(data)
View(data)

# VARIABLE DESCRIPTIONS:
#  survival        Survival
# (0 = No; 1 = Yes)
# pclass          Passenger Class
# (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
# (C = Cherbourg; Q = Queenstown; S = Southampton


###############################
## Explorative Data Analysis ##
###############################

####################
## Visualisations ##
####################

library(ggplot2)

# Bar Chart of Survived
# change survived to a categorical variable
# with levels 0 and 1
data$Survived <- as.factor(data$Survived)
levels(data$Survived)
ggplot(data = data, mapping = aes(x = Survived, fill = Survived)) + geom_bar(fill = c("red", "green")) + labs(title = "Bar Chart of Survival", x = "Survived", y = "Count") 
# we can see that the data is not comptelely balanced
# there are significantly more death

# Bar Chart of Class
ggplot(data, aes(x = Pclass)) + geom_bar() + labs(title = "Bar Chart of Class", x = "Class", y = "Count")
# we can see a lot more people in 3rd Class
# Bar Chart of Class vs Survied
ClassSurvived <- paste(data$Pclass, data$Survived)
ggplot(data = data, aes(x = ClassSurvived)) + geom_bar() + labs(title = "Bar Chart of Class Survived", x = "Class Survived", y = "Count")
# we can see that the vast majority of 3 class people did not survive

# Bar Chart of Sex
ggplot(data, aes(x = Sex, fill = Sex)) + geom_bar(fill = c("pink", "blue")) + labs(title = "Bar Chart of Sex", x = "Sex", y = "Count")
# we can see that there were more men than women on the titanic
SexSurvived <- paste(data$Sex, data$Survived)
ggplot(data, aes(SexSurvived)) + geom_bar() + labs(title = "Bar Chart of Sex Survived", x = "Sex Survived", y = "Count")
# we can see from the bar chart that the vast majority of males did not survive

# Bar Chart of Pclass
ggplot(data, aes(x = Pclass)) + geom_bar() + labs(title = "Bar Chart of Class", x = "Class", y = "Count")
# we can see that there were more people in third class on the titanic
PclassSurvived <- paste(data$Pclass, data$Survived)
ggplot(data, aes(PclassSurvived)) + geom_bar() + labs(title = "Bar Chart of Class Survived", x = "Class Survived", y = "Count")
# we can see from the bar chart that the vast majority of third class did not surviv

# Bar Chart of SibSp
ggplot(data, aes(x = SibSp)) + geom_bar() + labs(title = "Bar Chart of Siblings", x = "Siblings", y = "Count")
# we can see that most people did not have siblings on the titanic
SibSpSurvived <- paste(data$SibSp, data$Survived)
ggplot(data, aes(SibSpSurvived)) + geom_bar() + labs(title = "Bar Chart of Siblings Survived", x = "Siblings Survived", y = "Count")
# we can see from the bar chart that the vast majority of people without siblings did not survive

# Histogram of Age
?geom_histogram
ggplot(data, aes(x = Age)) + geom_histogram(binwidth = 5) + labs(title = "Histogram of Age", x = "Age", y = "Frequency")
# we can see that the distribution of Age is slightly right skewed
# note we have 177 rows with NA values for Age

# Histogram of Fare
?geom_histogram
ggplot(data, aes(x = Fare)) + geom_histogram(binwidth = 5) + labs(title = "Histogram of Fare", x = "Fare", y = "Frequency")
# we can see that the distribution of Fare is very right skewed
# note that the Age and Fare are not on the same measurement scale (???)
# Fare could possible distort out the signifcance of Age
# possible Standardisation needed???

############################
## Descriptive Statistics ##
############################

# Age
summary(Age)
range(Age)

# Fare
summary(Fare)
range(Fare)
typeof(Fare)

# parch
summary(Parch)
typeof(Parch)

# SibSp
summary(SibSp)

# Pclass
summary(Pclass)

# Embarked
summary(Embarked)

# Survived 
summary(Survived)

######################
## Loading Test Set ##
######################

test <- read.csv(file = "test.csv")
ncol(test)
View(test)

# test passenger id ranges from 892 to 1309

str(test)
summary(test)

####################################
## Combine Test Set and Train Set ##
####################################

# doing this save us having to process the test data set

# create empty survived vector for test set
TestSurvived <- rep(x = NA, times = 418)
test <- cbind(TestSurvived, test)
names(test)[names(test) == "TestSurvived"] <- "Survived"
# row bind data and test
data <- rbind(data, test)

#######################
## Feature Selection ##
#######################

# Option 1:remove features
# Option 2: keep all features

# Select the appropriate features

# Option 1: Remove features
# Name would not determine the survival of a person
# ticket would not effect survival rate
# what type of variable is ticket
# cabin would not effect survival rate
# cabin is missing a lot of data
data <- data[ , c(2, 3, 5, 6, 7, 8, 10, 12)]
head(data)
tail(data)

# Option 2: Keep all features

summary(data)
str(data)

########################
## Feature Processing ##
########################

##########
## Name ##
##########

library(stringi)

# Create a character vector of name
Name <- as.character(data$Name)

# Search each for an appropriate title and replace name with said title
Name[which(stri_detect_fixed(str = Name, pattern = " Capt. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Col. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Major. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Dr. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Rev. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Jonhkeer. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Sir. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Don. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Countess. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Dona. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Master. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Lady. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Mrs. "))] <- "Mrs"
Name[which(stri_detect_fixed(str = Name, pattern = " Mme. "))] <- "Mrs"
Name[which(stri_detect_fixed(str = Name, pattern = " Mlle. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Ms. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Miss. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Mr. "))] <- "Mr"

# update the data
data$Name <- Name

# rename Name to Title
names(data)[names(data) == "Name"] <- "Title"

# turn title in to a factor
data$Title <- as.factor(x = data$Title)

##############
## Embarked ##
##############

# Notice that there are two missing values in embarked
Embarked <- data$Embarked
Embarked <- as.factor(x = Embarked)
levels(data$Embarked)
which(data$Embarked == "")

# we have a few options
# Option 1: remove these values
# Option 2: impute values
# Option 3: leave the values

# Option 2
# as embarked is a factor
# we can impute values using regression or assign the most common 

# assigning the most common factpr
summary(data$Embarked)
Embarked[which(Embarked == "")] <- "S"

# update the factor levels
Embarked <- factor(Embarked)
levels(Embarked)

# update the data frame
data$Embarked <- Embarked
levels(data$Embarked)

###########
## Cabin ##
###########

# notice cabin has missing levels as well
summary(Cabin)
Cabin <- data$Cabin
levels(x = Cabin)

# create a new level called "unknown"
levels(x = Cabin) <- c("Unknown", levels(x = Cabin))
levels(x = Cabin)


# add in unknown level
Cabin[which(Cabin == "")] <- "Unknown"
summary(Cabin)

# update factor levels to remove " " level
Cabin <- factor(Cabin)
levels(Cabin)
data$Cabin <- Cabin
levels(data$Cabin)

#########
## Sex ##
#########

data$Sex <- as.factor(x = data$Sex)

###########
## Parch ##
###########

# change parch to a categorical variable
# with seven levels, 0 -> 6
data$Parch <- as.factor(data$Parch)
levels(data$Parch) <- 0:9
levels(data$Parch)

###########
## SibSp ##
###########

# change SibSp into a categorical variable
# with nine levels, 0 -> 8
data$SibSp <- as.factor(data$SibSp)
levels(data$SibSp) <- 0:8
levels(data$SibSp)

############
## Pclass ##
############

# change Pclass into a categorical variable
# with three levels, 1 -> 3
data$Pclass <- as.factor(data$Pclass)
levels(data$Pclass)

str(data)
summary(data)

#########
## Age ##
#########

# Age features missing data

#IMPORTANT NOTE: some classification algorithms cannot process missing values

# Two options 
# Option 1: remove na values from the data
# Option 2: impute values for the missing data using the average or regression
# Option 3: leave NA values in data set

# Option 1:remove the values

?complete.cases()
data <- data[complete.cases(data), ]
# Note: should have 714 observations

# Option 2: impute the values

# we can impute values using many methods such as
# the mean
# the mode
# the median
# regression

# NOTE: Not really a valid option in this case
# as we do not have any useful variables to determine the Age of a person
# i.e. 
# is Sex an indicator of Age?
# is Pclass an indicator of Age?
# is Fare an indicator of Age?
# is SibSp an indicator of Age?
# is Parch an indicator of Age?
# is Embarked an indicator of Age?
# is Survived an indicator of Age?

library("mice")

# impute missing values using mice() command
?mice()

# impute missing values for whole dataset using all features except Survived and Passengerid
# imputing missing values using predictions from linear regression 
imputedata <- data[ , -c(1, 2, 9)]
impAge <- mice(data = imputedata, m = 1, method = "norm.predict")
summary(impAge)
attributes(impAge)
# store the data
impdata <- complete(impAge)
data$Age <- impdata$Age
# Option 3: Leave NA values in the dataset
# some calssification algorithms like random forests work with NA values

###########################
## Partitioning the data ##
###########################

# ALL VALUES INCLUDED
# separate the test data from the training data
# testing set
test <- data[((nrow(data) - 418) + 1):nrow(data), -c(1,2)]
# training set
data <- data[1:(nrow(data) - 418), ]
# further divide the data into training and validation sets
train <- data[1:(nrow(data) * 0.7), -1 ]
valid <- data[(nrow(data) * 0.7):nrow(data), -c(1, 2)]
validlabels <- data[(nrow(data) * 0.7):nrow(data), c(1, 2)]


# note the length of the data
# depending on the options taking this will vary significantly

# Option 1: Partition 7:3
# Option 2: Partition 6:4

# Option 1: Partition 7:3
# training set
train <- data[1:(nrow(data) * 0.7), ]
# validation set
valid <- data[(nrow(data) * 0.7):nrow(data), -1]
# validation set labels
validlabels <- data[(nrow(data) * 0.7):nrow(data), 1]

# Option 2: Partition 6:4
# training set
train <- data[1:(nrow(data) * 0.6), ]
# validation set
valid <- data[(nrow(data) * 0.6):nrow(data), -1]
# validation set labels
validlabels <- data[(nrow(data) * 0.6):nrow(data), 1]

#######################
## Sampling the Data ##
#######################

library(ROSE)

?ovun.sample()
# NOTE: ovun.sample orders the data

data$Survived <- as.factor(data$Survived)
levels(data$Survived) <- 0:1
levels(data$Survived)
summary(data$Survived)


# three options
# (1) over sample the data - add no observations to balance distribtuion
# (2) under sample the data - remove yes observations to balance distribution
# (3) mixture sample the data - mixture of under and over sampling

# Option 1: Over sample the data
balanced_data <- ovun.sample(Survived ~ ., data = data, method = "over", na.action = "na.pass")
data <- balanced_data$data
# Bar Chart of Survived
ggplot(data = data, mapping = aes(x = Survived, fill = Survived)) + geom_bar(fill = c("red", "green")) + labs(title = "Bar Chart of Survival", x = "Survived", y = "Count") 
# we can see that the data is now roughly balanced
summary(data$Survived)

# Option 2: Under sample the data
balanced_data <- ovun.sample(Survived ~ ., data = data, method = "under", na.action = "na.pass")
data <- balanced_data$data
# Bar Chart of Survived
ggplot(data = data, mapping = aes(x = Survived, fill = Survived)) + geom_bar(fill = c("red", "green")) + labs(title = "Bar Chart of Survival", x = "Survived", y = "Count") 
# we can see that the data is now roughly balanced
summary(data$Survived)

# sample training data
balanced_train <- ovun.sample(Survived ~ ., data = train, method = "over", na.action = "na.pass")
train <- balanced_train$data

################################
## Standardising Numeric Data ##
################################

# Two Options
# Option 1: standardise data to be on same scale
# Option 2: don't standardise data

# Option 1

# standardise Age and Fare to have mean 0 and standard deviation 1
?scale()
# Age
data$Age <- scale(data$Age)
round(mean(data$Age, na.rm = T))
var(data$Age, na.rm = T)
# Fare 
data$Fare <- scale(data$Fare)
round(mean(data$Fare, na.rm = T))
var(data$Fare, na.rm = T)

# Option 2

# standardising isn't necessary as Age and Fare are roughly on the same scale

# see an update of the data so far
summary(data)
str(data)

########################
## Randomise the Data ##
########################

head(data)
# currently the data is ordered due to ovun.sample() command
# as such we need to randomise the data
set.seed(seed = 1234)
u <- runif(n = nrow(data))
data <- data[order(u), ]

head(data)

View(data)

# randomise train data
u <- runif(n = nrow(train))
train <- train[order(u), ]


#######################################
## Classification Models with rminer ##
#######################################

library(rminer)

library(caret)

# fit the following models
# (1) Decision Tree
# (2) Random Forests
# (3) Naive Bayes Classifiers
# (4) K-Nearest Neighbours
# (5) Support Vector Machines
# (6) Logistic Regression
# (7) Conditional Inference Trees

View(data)

#########################
## Decision Tree Model ##
#########################

#NOTE: decision trees can handle missing data

# fit decision tree model using fit()
?fit()
dtree <- fit(Survived ~ ., data = train, model = "rpart", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = dtree, newdata = valid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, validlabels))

# measure input importance 
idtree <- Importance(M = dtree, data = train)
idtree$value
# numeric vector with the computed sensitivity analysis measure for each variable
idtree$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance


#########################
## Random Forest Model ##
#########################

# NOTE: Random Forests cannot handle missing data

# fit random forests model using fit()
?fit()
rforest <- fit(Survived ~ ., data = train, model = "randomForest", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = rforest, newdata = valid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, validlabels))

# measure input importance 
irforest <- Importance(M = rforest, data = train)
irforest$value
# numeric vector with the computed sensitivity analysis measure for each variable
irforest$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

#############################
## Naive Bayes Classifiers ##
#############################

# NOTE: Naive Bayes Classifiers can handle missing data

# fit Naive Bayes Classifiers model using fit()
?fit()
nbayes <- fit(Survived ~ ., data = train, model = "naive", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = nbayes, newdata = valid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, validlabels))

# measure input importance 
inbayes <- Importance(M = nbayes, data = train)
inbayes$value
# numeric vector with the computed sensitivity analysis measure for each variable
inbayes$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

##########################
## K-Nearest Neighbours ##
##########################

# Knn cannot handle missing data

# fit K-Narest Neighbours model using fit()
?fit()
knnm <- fit(Survived ~ ., data = train, model = "knn", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = knnm, newdata = valid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, validlabels))

# measure input importance 
iknnm <- Importance(M = knnm, data = train)
iknnm$value
# numeric vector with the computed sensitivity analysis measure for each variable
iknnm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

#############################
## Support Vector Machines ##
#############################

# Support Vector Machines cannot handle missing data

# fit support vector machines model using fit()
?fit()
svmm <- fit(Survived ~ ., data = train, model = "svm", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = svmm, newdata = valid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, validlabels))

# measure input importance 
isvmm <- Importance(M = svmm, data = train)
isvmm$value
# numeric vector with the computed sensitivity analysis measure for each variable
isvmm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

#########################
## Logistic Regression ##
#########################

# logistic regression can handle missing data

# fit logistic regression model using fit()
?fit()
lr <- fit(Survived ~ ., data = train, model = "lr", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = lr, newdata = valid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, validlabels))

# measure input importance 
ilr <- Importance(M = lr, data = train)
ilr$value
# numeric vector with the computed sensitivity analysis measure for each variable
ilr$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

#################################
## Conditional Inference Trees ##
#################################

# logistic regression can handle missing data

# fit logistic regression model using fit()
?fit()
ctreem <- fit(Survived ~ ., data = train, model = "ctree", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = ctreem, newdata = valid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, validlabels))

# measure input importance 
ictreem <- Importance(M = ctreem, data = train)
ictreem$value
# numeric vector with the computed sensitivity analysis measure for each variable
ictreem$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

###################################
## Model Results and Comparisons ##
###################################

# Decisions
# (1) Oversample the Data
# (2) Do not standardise the data
# (3) Data will be partitioned 7:3
# (4) Include and exclude NA's depending on context, not imputing values

# Model                         Accuracy    
# Decision Trees                0.8077 
# Random Forests                0.8577
# Naive Bayes Classifiers       0.5851
# K-Nearest Neighbours          0.8115
# Support Vector Machines       0.8308
# Logistic Regression           0.8192
# Conditional Inference Trees   0.7808
#

#######################################
## Note on handling the missing data ##
#######################################

# Option 1: impute values for the missing data
# Option 2: use a model that does handles missing data

######################
## Loading Test Set ##
######################

test <- read.csv(file = "test.csv")
ncol(test)
View(test)

str(test)
summary(test)

#############################
## Processing the Test Set ##
#############################

View(test)

# (1) feature selection
test <- test[ , c(2, 4, 5, 6, 7, 9, 11)]

# (2) feature processing
# change parch to a categorical variable
# with ten levels, 0 -> 9
test$Parch <- as.factor(test$Parch)
levels(x = test$Parch) <- 0:9
levels(x = test$Parch)
# change SibSp into a categorical variable
# with nine levels, 0 -> 8
test$SibSp <- as.factor(test$SibSp)
levels(x = test$SibSp) <- 0:8
levels(x = test$SibSp)
# change Pclass into a categorical variable
# with three levels, 1 -> 3
test$Pclass <- as.factor(test$Pclass)
levels(x = test$Pclass)
# check embarked
test$Embarked <- as.factor(test$Embarked)
levels(test$Embarked) <- c("", "C", "Q", "S")
levels(test$Embarked)

# (3) Processing NA values
# we are required to give answers for all 418 observations
# as such, we can not remove any observations
# even for purpose of model building?

# (4) No need to sample the data

# (5) Standardise the numeric data
# Note data can still be standardised even with NA's repsent
# Age
test$Age <- scale(test$Age)
round(mean(test$Age, na.rm = T))
var(test$Age, na.rm = T)
# Fare 
test$Fare <- scale(test$Fare)
round(mean(test$Fare, na.rm = T))
var(test$Fare, na.rm = T)

# (6) Randomisation is not necessary

# (7) Partitioning the data is not necessary

#############################
## Predicting the Test Set ##
#############################

View(data)
View(test)

dtreepred <- predict(object = dtree, newdata = test, type = "class")
rforestpred <- predict(object = rforest, newdata = test, type = "class")
nbayespred <- predict(object = nbayes, newdata = test, type = "class")
knnmpred <- predict(object = knnm, newdata = test, type = "class")
svmmpred <- predict(object = svmm, newdata = test, type = "class")
lrpred <- predict(object = lr, newdata = test, type = "class")
ctreempred <- predict(object = ctreem, newdata = test, type = "class")

# Step 1: Remove NA during training and run the best model to predict for entire test set, bar the 87 missing values. (classify entire test set, except the 87 NA values)
# Step 2: Keep NA values during training and run deicion tree and logistic regression. (classify the 87 NA values)
# Step 3: Remove NA values during training and run 4 models parallel. (classify for certain values)


# Step 1

# create vector of indexes
testpred <- rforestpred
testpred

# Step 2

# create vector of indexes
j <- which(is.na(x = testpred))
testpred[j] <- dtreepred[j]
testpred

# Step 3

# create vector of indexes
i1 <- which(dtreepred == rforestpred)
i2 <- which(svmmpred == knnmpred)
i <- intersect(i1, i2)
length(rforestpred[i])

# insert predictions by index
testpred[i] <- rforestpred[i]
testpred

##########################
## Writing to CSV files ##
##########################

?write.table()
write.table(testpred, file = "testpred.csv", sep = ",")

##############
## Appendix ##
##############

#################################
## Imputing the missing values ##
#################################

# Is this a logical move?
# Can we predict age with the other variables
# Maybe fare is an indication of age
AgeFare <- paste(Age, Fare)
ggplot(data = data, aes(AgeFare)) + geom_bar() + labs(title = "Bar Chart of AgeFare", x = "Age Fare", y = "Count")

?cor()
cor(data$Age, data$Fare)
plot(Age, Fare)
# Theres does not appear to be an association between age and fare

# Chi-Squared Test for independence
# between two catehorcial variables
# NOTE: Fare and Age are not categorical variables NOTE VALID TEST
# Ho: There is no relation between Age and Fare (=) Age and Fare independent
# Ha: There is a relation between Age and Fare (=) Age and Fare not independent
?chisq.test()
chisq.test(x = Age, y = Fare)
round(2.235*exp(-09))
# As p-value is approx eqiual to 0
# we can conclude that there is some sort of a relationship between Age and Fare

chisq.test(x = Age, y = Sex) # not significant
chisq.test(x = Age, y = Pclass) # significant
chisq.test(x = Age, y = SibSp) # significant
chisq.test(x = Age, y = Parch) # significant
chisq.test(x = Age, y = Embarked) # significant

##############################
## Prediciting the Test Set ##
##############################

# create empty numeric vector of length 418 for test predicitons
testpred <- rep(x = NA, times = 418)
# turn testpred into a factor variable with 2 levels; 0, 1
testpred <- as.factor(testpred)
levels(testpred) <- c(0, 1)
testpred