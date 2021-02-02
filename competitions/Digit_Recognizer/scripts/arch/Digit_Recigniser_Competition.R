#######################################################################################################################
## Packages ###########################################################################################################
#######################################################################################################################

library(nortest)
library(MASS)
library(ridge)
library(acepack)
library(plot3D)
library(car)
library(dummy)
library(plyr)
library(glmnet)
library(moments)
library(formula.tools)
library(stringi)
library(rminer)
library(mda)
library(ggplot2)
library(pls)
library(mice)
library(nnet)
library(kernlab)
library(FactoMineR)
library(readr)
library(h2o)

#######################################################################################################################
## Digit Recogniser Competition #######################################################################################
#######################################################################################################################

getwd()
setwd(dir = "C:/Users/Margaret/Documents/Oisin/R/Kaggle Competitions/Digit Recognizer")
train <- read.csv(file = "train.csv")
test <- read.csv(file = "test.csv")
# Consolidate data into one data frame
# NOTE: remove ID and y attributes
full_data <- rbind(train[,-1], test)
head(train)

# can use R package readr to load rectangular data, such as csv files fast
train <- read_csv(file = "train.csv")
test <- read_csv(file = "test.csv")

########################################################################################################################
## Visual Image ########################################################################################################
########################################################################################################################

# lets have a look at the first images
im1 <- matrix(unlist(train[1,-1]), nrow = 28, byrow = T)
image(x = im1, col = grey.colors(255))
# the second image
im2 <- matrix(unlist(train[2,-1]), nrow = 28, byrow = T)
image(x = im2, col = grey.colors(255))
# the third image
im3 <- matrix(unlist(train[3,-1]), nrow = 28, byrow = T)
image(x = im3, col = grey.colors(255))
# the four image
im4 <- matrix(unlist(train[4,-1]), nrow = 28, byrow = T)
image(x = im4, col = grey.colors(255))

# rotate 

########################################################################################################################
## Principle Components Analysis #######################################################################################
########################################################################################################################

?PCA()
principle_components_analysis <- PCA(X = full_data)
summary(principle_components_analysis)
principle_components_analysis$ind$coord

#########################################################################################################################
## Model Building #######################################################################################################
#########################################################################################################################

## start a local h2o cluster
localH2O = h2o.init(max_mem_size = '6g', # use 6GB of RAM of *GB available
                    nthreads = -1) # use all CPUs (8 on my personal computer :3)

train[,1] = as.factor(train[,1]) # convert digit labels to factor for classification
train_h2o = as.h2o(train)

test_h2o = as.h2o(test)

## set timer
s <- proc.time()

## train model
model =
  h2o.deeplearning(x = 2:785,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(100,100), # two layers of 100 nodes
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 15) # no. of epochs

## print confusion matrix
h2o.confusionMatrix(model)

## print time elapsed
s - proc.time()

## classify test set
h2o_y_test <- h2o.predict(model, test_h2o)

## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "submission-r-h2o.csv", row.names=F)

## shut down virutal H2O cluster
h2o.shutdown(prompt = F)
