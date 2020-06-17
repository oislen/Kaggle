#######################################################################################################################
## Packages ###########################################################################################################
#######################################################################################################################

library(BuenaVista)

library(nortest)
library(ridge)
library(acepack)
library(plot3D)
library(car)
library(plyr)
library(moments)
library(formula.tools)
library(stringi)
library(rminer)
library(pls)
library(nnet)
library(kernlab)
library(FactoMineR)

######################################################################################################################
## House Prices Competition ##########################################################################################
######################################################################################################################

# NOTES
# Steps
# Profile the data
# Clean the data
# Use forward elimination to identify useful variables
# iteratively fit the varaibles
# for linear regression remember interaction terms, power terms, ratios
# use advanced regression techniques such as Ridge Regression, LASSO and Elastic Nets
# try ensemble modelling such as bagging, boosting and stacking
# try gradient boosting and extreme gradient boosting

getwd()
setwd(dir = "C:/Users/Margaret/Documents/Oisin/R/Kaggle Competitions/House Prices")
train <- read.csv(file = "train.csv")
test <- read.csv(file = "test.csv")
# Consolidate data into one data frame
SalePrice <- rep(NA, times = 1459)
test1 <- cbind(test, SalePrice)
full_data <- rbind(train, test1)

######################################################################################################################
## Data Profiling / Exploratory Data Analysis ########################################################################
######################################################################################################################

# we have a lot of missing data in the attributes
# sapply statistics
# number of missing values per attribute
sapply(fulldata, function(x) sum(is.na(x)))
# missing value percentage
sapply(fulldata, function(x) sum(is.na(x))*100/nrow(fulldata))
# attribute is a character variable
sapply(fulldata, function(x) is.character(x))
# attribute is a numeric variable
sapply(fulldata, function(x) is.numeric(x))
# attribute is a categorical / factor variable
sapply(fulldata, function(x) is.factor(x))

##############
## Metadata ##
##############

# code up a function that returns metadata information

####################
## Visualisations ##
####################

visualise_data <- function(dataset) {
  # function that automatically plots appropriate visualisations for a given dataset using the ggplot2 package
  library(ggplot2)
  # plot a histogram for continuous variables and a barchart for numeric varaibles
  # NOTE: ggplot automatically removes missing data from visualised attributes
  for (i in 1:ncol(dataset)) {
    vname <- colnames(dataset)[i]
    if (is.factor(dataset[,i])){
      # Categorical Variables
      # Bar Chart
      print(ggplot(data = dataset, mapping = aes(x = dataset[,i])) + geom_bar() + labs(title = paste("Bar Chart of", vname, sep = " "), x = vname, y = "Count")) 
    } else {
      # Continuous Variables
      # Histogram
      print(ggplot(data = dataset, mapping = aes(x = dataset[,i])) + geom_histogram() + labs(title = paste("Histogram of", vname, sep = " "), x = vname, y = "Frequency"))
      # Boxplot
      print(ggplot(data = dataset, mapping = aes(y = dataset[,i], x = "")) + geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 2) + labs(title = paste("Boxplot of", vname, sep = " "), x = "", y = vname))
    } 
  }
}

# visualise_data(dataset = train)
visualise_data(dataset = full_data)
# visualise_data(dataset = reduced_data)

############################
## Descriptive statistics ##
############################

# DESCRIPTIVE STATISTICS FOR CATEGORICAL VARIABLES
factor_descriptive_statistics <- function(dataset) {
  # function that automatically prints relevant descriptive statistics for attributes in a given dataset
  library(moments)
  l = 1 # row index for categorical variables
  # create the data frame to hold the categorical descriptive statistics
  factor_num_col <- 8
  factor_num_row <- sum(sapply(X = dataset, FUN = function(x) is.factor(x)))
  factor_descriptive_statistics <- as.data.frame(matrix(nrow = factor_num_row, ncol = factor_num_col))
  colnames(factor_descriptive_statistics) <- c("1st_mode_name", "1st_mode_count", "1st_mode_percent", "2nd_mode_name", "2nd_mode_count", "2nd_mode_percent", "NA_count", "NA_percent")
  for (i in 1:ncol(dataset)) {
    if (is.factor(dataset[,i])){
      # first write in the row name i.e. the attribute name
      rownames(factor_descriptive_statistics)[l] <- colnames(dataset)[i]
      # next compute the descriptive statistics for each cell in the row
      factor_descriptive_statistics[l, 1] <- names(which.max(summary(dataset[,i])))
      factor_descriptive_statistics[l, 2] <- summary(dataset[, i])[which.max(summary(dataset[,i]))]
      factor_descriptive_statistics[l, 3] <- summary(dataset[, i])[which.max(summary(dataset[,i]))] * 100 / sum(summary(dataset[,i]))
      factor_descriptive_statistics[l, 4] <- names(sort(summary(dataset[, i]), decreasing = T)[2])
      factor_descriptive_statistics[l, 5] <- sort(summary(dataset[, i]), decreasing = T)[2]
      factor_descriptive_statistics[l, 6] <- sort(summary(dataset[, i]), decreasing = T)[2] * 100/ sum(summary(dataset[,i]))
      factor_descriptive_statistics[l, 7] <- length(which(is.na(dataset[,i])))
      factor_descriptive_statistics[l, 8] <- length(which(is.na(dataset[,i]))) * 100 / sum(summary(dataset[,i]))
      # print(paste("Categorical Attribute:", colnames(dataset)[i], sep = " "))
      # print(summary(dataset[,i]))
      l = l + 1
    } 
  }
  return(factor_descriptive_statistics)
}
# NOTE: could code up a table of descriptive statistics for continuous and decrete variables, metadata
# descriptive_statistics(dataset = train)
# factor_descriptive_statistics(dataset = reduceddata)
full_data_factor_descriptive_statistics_df <- factor_descriptive_statistics(dataset = full_data)

# DESCRIPTIVE STATISTICS FOR CONTINUOUS FEATURES
numeric_descriptive_statistics <- function(dataset) {
  # IMPORTANT NOTE: all the descriptive statistics are calculated with the NA values removed
  # function that automatically prints relevant descriptive statistics for attributes in a given dataset
  library(moments)
  k = 1 # row index for numeric variables
  # create the data frame to hold the numeric descriptive statistics
  numeric_num_col <- 14
  numeric_num_row <- sum(sapply(X = dataset, FUN = function(x) is.numeric(x)))
  numeric_descriptive_statistics <- as.data.frame(matrix(nrow = numeric_num_row, ncol = numeric_num_col))
  colnames(numeric_descriptive_statistics) <- c("mean", "variance", "standard deviation", "min", "max", "range", "1st_Qu", "median", "3rd Qu", "IQR", "skewness", "kurtosis", "NA_count", "NA_percent")
  for (i in 1:ncol(dataset)) {
    if (is.numeric(dataset[,i])) {
      # first write in the row name i.e. the aattribute name
      rownames(numeric_descriptive_statistics)[k] <- colnames(dataset)[i]
      # next compute the descriptive statistics for each cell in the row
      numeric_descriptive_statistics[k, 1] <- mean(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 2] <- var(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 3] <- sd(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 4] <- min(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 5] <- max(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 6] <- max(dataset[,i], na.rm = T) - min(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 7] <- quantile(x = dataset[,i], probs = 0.25, na.rm = T)
      numeric_descriptive_statistics[k, 8] <- median(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 9] <- quantile(x = dataset[,i], probs = 0.75, na.rm = T)
      numeric_descriptive_statistics[k, 10] <- quantile(x = dataset[,i], probs = 0.75, na.rm = T) - quantile(x = dataset[,i], probs = 0.25, na.rm = T)
      numeric_descriptive_statistics[k, 11] <- skewness(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 12] <- kurtosis(dataset[,i], na.rm = T)
      numeric_descriptive_statistics[k, 13] <- length(which(is.na(dataset[,i])))
      numeric_descriptive_statistics[k, 14] <- length(which(is.na(dataset[,i]))) * 100 / nrow(dataset)
      
      # print(paste("Numeric Attribute:", colnames(dataset)[i], sep = " "))
      # print(summary(dataset[,i]))
      # print(paste("Interquartile Range", IQR(dataset[,i], na.rm = T), sep = " -> "))
      # print(paste("Variance", var(dataset[,i], na.rm = T),  sep = " -> "))
      # print(paste("Standard Deviation", sd(dataset[,i], na.rm = T),  sep = " -> "))
      # print(paste("Skeweness", skewness(dataset[,i], na.rm = T), sep = " -> "))
      # print(paste("Kurtosis", kurtosis(x = dataset[,i], na.rm = T), sep = " -> "))
      k = k + 1
    }
    # print("############################")
  }
  return(numeric_descriptive_statistics)
}


full_data_numeric_descriptive_statistics_df <- numeric_descriptive_statistics(dataset = full_data)

#################################
## Correlation Plots and Tests ##
#################################

correlation_analysis <- function(dataset, response_variable, response_name) {
  # function that automatically constructs relevant correlation plots and tests from a given dataset
  # NOTE: response_variable must be an attribute in the given dataset
  # i.e. specify of the form dataset$repsonse_variable
  # repsonse name should be a character object / string denoting the name of the response variable
  for (i in 1:ncol(dataset)) {
    if (is.numeric(dataset[,i])){
      vname <- colnames(dataset)[i]
      title <- as.character(paste("Correlation:", response_name, "and", vname, sep = " "))
      print(title)
      print(cor.test(x = response_variable, y = dataset[,i], method = "pearson", alternative = "two.sided"))
      # Ho: variables X and Y are not correlated
      # Ha: variables X and Y are correlated
      plot(x = dataset[,i], y = response_variable, main = title, xlab = vname, ylab = response_name)
      # plots y vs x, where y is the response_variable and x is the correlated numeric variable
      print("###############################")
    }
  }
}

correlation_analysis(dataset = train, response_variable = train$SalePrice, response_name = "SalePrice")

######################
## Outlier Analysis ##
######################
outlier_analysis <- function(dataset) {
  # function that saves the index of outliers from each variable of a given dataset 
  # create a data frameto hold the idenfitied outliers
  num_col <- sum(sapply(X = dataset, FUN = function(x) is.numeric(x)))
  outliers_data_frame <- as.data.frame(matrix(nrow = nrow(dataset), ncol = num_col))
  k = 1
  for ( i in 1:ncol(dataset)) {
    # outliers are only valid for continuous variables
    if (is.numeric(dataset[,i])) {
      print(paste(colnames(dataset)[i], "Outliers:", sep = " "))
      # outliers either lie below Q1 - 1.5 * IQR or above Q3 - 1.5 * IQR
      lower_outlier_bound <- quantile(x = dataset[,i], na.rm = T)[2] - 1.5 * IQR(x = dataset[,i], na.rm = T) 
      upper_outlier_bound <- quantile(x = dataset[,i], na.rm = T)[4] + 1.5 * IQR(x = dataset[,i], na.rm = T) 
      print(which(dataset[,i] > upper_outlier_bound | dataset[,i] < lower_outlier_bound))
      # print(dataset[which(dataset[,i] > upper_outlier_bound | dataset[,i] < lower_outlier_bound), i])
      print("###############################")
      variable_outliers <- which(dataset[,i] > upper_outlier_bound | dataset[,i] < lower_outlier_bound)
      padding <- rep(x = NA, times = (nrow(dataset) - length(variable_outliers)))
      data <- c(variable_outliers, padding)
      outliers_data_frame[,k] <- data
      colnames(outliers_data_frame)[k] <- colnames(dataset)[i]
      k = k + 1
    }
  }
  return(outliers_data_frame)
}

full_data_outliers <- outlier_analysis(dataset = full_data)

#######################################################################################################################
## Data Processing ####################################################################################################
#######################################################################################################################

# Steps
# (1) Remove Useless features
# (2) Impute missing valeus
# (3) Feature Engineering
# (4) Transformations
# (5) Standardisations
# (6) Dummy Encoding Categorical Variables


#######################
## Feature Selection ##
#######################

# remove ID and SalePrice attributes
reduced_data <- full_data[,-c(1, 81)]

# removing attributes with too many missing values
remove_attribute_with_missing_data <- function(data, percentage = 20) {
  # function that automatically remove attributes with too many missing observations (default is 20% missing observations)
  # print data variables with too many missing values
  reduceddata <- data[,1]
  for(i in 1:(ncol(data))) {
    # for all columns in the specified dataset do
    if (sum(is.na(x = data[,i]))*100/nrow(data) < percentage) {
      # keep an attribute if and only if it is not missing less than percentage (default = 20) of its data 
      reduceddata <- as.data.frame(cbind(reduceddata, data[,i]))
      colnames(reduceddata)[1] <- colnames(data)[1]
      colnames(reduceddata)[ncol(reduceddata)] <- colnames(data)[i]
    } else {
      # print the attribute that was removed from the dataset
      print(paste(colnames(data)[i], "attribute removed, missing over", as.character(percentage), "percent of its observations", sep = " "))
    }
  }
  # return the created reduced dataset as a data frame
  return(reduceddata <- reduceddata[,-1])
}

# reduced_data <- remove_attribute_with_missing_data(data = fulldata, percentage = 2)
# reduced_data <- remove_attribute_with_missing_data(data = fulldata, percentage = 10)
reduced_data <- remove_attribute_with_missing_data(data = reduced_data, percentage = 20)
# summary(reduced_data)
# sapply(reduced_data, function(x) sum(is.na(x)))
sapply(reduced_data, function(x) sum(is.na(x))*100/nrow(reduced_data))

# manually remove data with too many missing values
sapply(full_data, function(x) sum(is.na(x))*100/nrow(fulldata))[c(4,7,31,32,33,34,36,58,59,60,61,64,65,73,74,75)]
reduced_data <- fulldata[,-c(4,7,31,32,33,34,36,58,59,60,61,64,65,73,74,75)]
sapply(reduced_data, function(x) sum(is.na(x)))
# missing value percentage
sapply(full_data, function(x) sum(is.na(x))*100/nrow(fulldata))


######################
## Data Imputations ##
######################

# imputing mean or mode 
mean_mode_imputation <- function(dataset, percentage) {
  # function automatically imputes the mean and mode for attributes in a given dataset
  # the percentage specifies identifies the attributes with a missing number of observations
  # the missing observations in these identified attributes are then imputed using the mean or mode
  for (i in 1:ncol(dataset)) {
    # for each column in the specified dataset
    vname <- colnames(dataset)[i]
    # retrieve the column name
    percentage_missing_data <- (sum(is.na(x = dataset[,i])) * 100) / nrow(dataset) 
    if (0 < percentage_missing_data && percentage_missing_data < percentage) {
      # variable is missing between 0 and percentage of its observations
      print(paste(vname, "is missing less than", as.character(percentage), "percent of its observerations", sep = " "))
      if (is.factor(dataset[,i])) {
        # if categorical varible impute the mean fro each missing observation
        mode <- attributes(summary(dataset[,i]))$names[which.max(summary(dataset[,i]))]
        print(paste("Impute Mode:", as.character(mode), sep = " "))
        dataset[which(is.na(dataset[,i])),i] <- mode
      } else {
        # if continuous varuable compute the mean for each missing observation
        mean <- mean(dataset[,i], na.rm = T)
        print(paste("Impute Mean:", as.character(mode), sep = " "))
        dataset[which(is.na(dataset[,i])),i] <- mean
      }
      print("######################################")
    }
  }
  return(dataset)
}

# reduced_data <- mean_mode_imputation(dataset = reduced_data, percentage = 2)
imputed_data <- mean_mode_imputation(dataset = reduced_data, percentage = 10)
# ridata <- mean_mode_imputation(dataset = reduced_data, percentage = 10)
# reduced_data <- mean_mode_imputation(dataset = reduced_data, percentage = 20)
sapply(imputed_data, function(x) sum(is.na(x))*100/nrow(imputed_data))

# imputation using prediction /  classification
impute_numeric_feature <- function(dataset, feature_index, method_used = c("norm.predict")) {
  library(mice)
  # NOTE: works best when imputing for one column / attribute
  # dataset: the specified dataset with missing values (all in 1 column) to impute
  # feature_index: the index of the feature you wish to impute that has the missing values in the dataset
  # Make sure the dataset does not an ID column
  # function imputes missing values for a numeric feature using linear regression
  # want to impute values for LotFrontage
  # impute missing values for whole dataset using all features except Survived and Passengerid
  # imputing missing values using predictions from linear regression 
  imputedata <- mice(data = dataset, method = method_used)
  # store the data
  impdata <- complete(imputedata)
  return(impdata)
}

# want to immpute lotfrintage which has column index 3
imputed_data <- impute_numeric_feature(dataset = imputed_data, feature = 3, method_used = "norm.predict")
# check missing values
sapply(imputed_data, function(x) sum(is.na(x))*100/nrow(imputed_data))

# divide into training and test datasets
# trainingset <- reduceddata[1:1460,-c(1)]
# testset <- reduceddata[1461:2919, -c(1)]

#################################
## Numeric Feature Engineering ##
#################################

# this section gives functions that:
# (1) extract the numeric attributes from a dataset
# (2) extract the categorical attributes from a dataset
# (3) derive power terms from numeric attributes in a dataset
# (4) interaction terms from numeric attributes in a dataset

# want to generate powers, interactions, ratios etc...
# specifically for the numeric variables
# we shall work on the numericvaraibles
head(numericvaraibles)

# EXTRACT NUMERIC DATA
extract_numeric_data <- function(dataset) {
  # This function extracts the numeric attributes out of a dataset and stores them in a seperate dataset
  # dataset: the specified dataset to extract the numeric attributes from
  # create a data frame to hold the numeric attributes
  num_row <- nrow(dataset)
  num_col <- sum(sapply(X = dataset, FUN = function(x) is.numeric(x)))
  numeric_data <- as.data.frame(matrix(nrow = num_row, ncol = num_col))
  # we will use a seperate index j to store the derived features in the power_data
  j = 1
  for (i in 1:ncol(dataset)){
    if(is.numeric(dataset[,i])) {
      numeric_data[,j] <- dataset[,i]
      colnames(numeric_data)[j] <- colnames(dataset)[i]
      j = j + 1
    }
  }
  return(numeric_data)
}

# numeric_data <- extract_numeric_data(dataset = reduceddata)
numeric_data <- extract_numeric_data(dataset = imputed_data)


# EXTRACT CATEGORICAL DATA
extract_factor_data <- function(dataset) {
  # This function extracts the categorical attributes out of a dataset and stores them in a seperate dataset
  # dataset: the specified dataset to extract the categorical attributes from
  # create a data frame to hold the categorical attributes
  num_row <- nrow(dataset)
  num_col <- sum(sapply(X = dataset, FUN = function(x) is.factor(x)))
  factor_data <- as.data.frame(matrix(nrow = num_row, ncol = num_col))
  # we will use a seperate index j to store the derived features in the power_data
  j = 1
  for (i in 1:ncol(dataset)){
    if(is.factor(dataset[,i])) {
      factor_data[,j] <- dataset[,i]
      colnames(factor_data)[j] <- colnames(dataset)[i]
      j = j + 1
    }
  }
  return(factor_data)
}
# factor_data <- extract_factor_data(dataset = reduceddata)
factor_data <- extract_factor_data(dataset = imputed_data)


# DERIVE POWER TERMS
derive_power_terms <- function(dataset, power, degree_term ) {
  # This function takes the numeric attributes out of a dataset, squares them and stores them in a seperate dataset
  # dataset: the specified dataset to derive power terms from
  # power: the power the terms should be raised to
  # degree_term: a character object that is used to name the newly derived terms
  # create a data frame to hold the squared attributes
  num_row <- nrow(dataset)
  num_col <- sum(sapply(X = dataset, FUN = function(x) is.numeric(x)))
  power_data <- as.data.frame(matrix(nrow = num_row, ncol = num_col))
  # we will use a seperate index j to store the derived features in the power_data
  j = 1
  for (i in 1:ncol(dataset)){
   if(is.numeric(dataset[,i])) {
     power_data[,j] <- dataset[,i]^power
     colnames(power_data)[j] <- paste(colnames(dataset)[i], degree_term, sep = "")
     j = j + 1
   }
  }
  return(power_data)
}
# squared_data <- derive_power_terms(dataset = reduceddata, power = 2, degree_term = "Sq")
squared_data <- derive_power_terms(dataset = numeric_data, power = 2, degree_term = "Sq")
# cubed_data <- derive_power_terms(dataset = reduceddata, power = 3, degree_term = "Cb")
# use stepwise selection of interaction data to find most useful terms
SalePrice <- train$SalePrice
Ltrainingset <- cbind(squared_data[1:1460,], SalePrice)
min_model <- lm(SalePrice ~ 1, data = Ltrainingset)
max_model <- formula(lm(SalePrice ~ ., data = Ltrainingset))
stp_model <- step(min_model, direction = 'both', scope = max_model)
summary(stp_model)
predict(stp_model, Ltestset)
# From Step-wise selection we have 27 interaction features that are useful
#SalePrice ~ OverallQualSq + GrLivAreaSq + GarageCarsSq + BsmtFullBathSq + 
#  MSSubClassSq + BsmtFinSF1Sq + YearBuiltSq + FireplacesSq + 
#  OverallCondSq + MasVnrAreaSq + X1stFlrSFSq + BsmtUnfSFSq + 
#  X2ndFlrSFSq + LotAreaSq + WoodDeckSFSq + ScreenPorchSq + 
#  YearRemodAddSq + TotRmsAbvGrdSq + BedroomAbvGrSq + KitchenAbvGrSq + 
#  LotFrontageSq + TotalBsmtSFSq + BsmtFinSF2Sq + GarageAreaSq + 
#  OpenPorchSFSq + FullBathSq + LowQualFinSFSq + BsmtHalfBathSq
squared_data <- subset(squared_data, select = c(GrLivAreaSq, GarageCarsSq, BsmtFullBathSq, MSSubClassSq, BsmtFinSF1Sq, YearBuiltSq, FireplacesSq, OverallCondSq, MasVnrAreaSq, X1stFlrSFSq, BsmtUnfSFSq, X2ndFlrSFSq, LotAreaSq, WoodDeckSFSq, ScreenPorchSq, YearRemodAddSq, TotRmsAbvGrdSq, BedroomAbvGrSq, KitchenAbvGrSq, LotFrontageSq, TotalBsmtSFSq, BsmtFinSF2Sq, GarageAreaSq, OpenPorchSFSq, FullBathSq, LowQualFinSFSq, BsmtHalfBathSq))
head(squared_data)

# DERIVE INTERACTION TERMS 
derive_interaction_terms <- function(dataset) {
  # DERIVE INTERACTION TERMS 
  # only wroks for dataset comprised of only numeric data
  # it will not work if there are categorcal attributes in the dataset
  # the number of interaction terms relative to the number of attributes follows the triangular sequence
  # for x = {2, 3, 4, 5, 6 etc ...}, f(x) = {1, 3, 6, 10, 15, 21 etc...}, where f(x) is the function for triangular numbers
  # 2 attributes = 1 interaction term
  # 3 attributes = 3 interaction terms
  # 4 attributes = 6 interaction terms
  # 5 attributes = 10 interaction terms
  # calculate the number of interaction terms
  triangular_numbers <- function(x) {
    num_interaction_terms <- x * (x + 1) / 2
    return(num_interaction_terms)
  }
  nth_term <- ncol(dataset) - 1
  num_interaction_terms <- triangular_numbers(x = nth_term)
  # create empty dataframe to store the interaction terms 
  num_row <- nrow(dataset)
  num_col <- num_interaction_terms
  interaction_data <- as.data.frame(matrix(nrow = num_row, ncol = num_col))
  colnames(interaction_data)
  k = 1
  for (i in 1:ncol(dataset)){
    j = i + 1
    v1name <- colnames(dataset)[i]
    #print(v1name)
    while(j <= ncol(dataset)){
      v2name <- colnames(dataset)[j]
      #print(paste(v1name, "*", v2name, sep = ""))
      #print(k)
      #print(j)
      interaction_data[,k] <- dataset[,i] * dataset[,j]
      colnames(interaction_data)[k] <- paste(v1name, "_x_", v2name, sep = "")
      j = j + 1
      k = k + 1
    }
  }
  return(interaction_data)
}
interaction_data <- derive_interaction_terms(dataset = numeric_data)
# stepwise selection of interaction data to find useful terms
SalePrice <- train$SalePrice
Ltrainingset <- cbind(interaction_data[1:1460,], SalePrice)
min_model <- lm(SalePrice ~ 1, data = Ltrainingset)
max_model <- formula(lm(SalePrice ~ ., data = Ltrainingset))
stp_model <- step(min_model, direction = 'both', scope = max_model)
summary(stp_model)
predict(stp_model, Ltestset)
# 57 noteable interaction terms
# SalePrice ~ OverallQual_x_GrLivArea, OverallQual_x_GarageCars, MasVnrArea_x_PoolArea, OverallCond_x_BsmtFinSF1, OverallQual_x_TotalBsmtSF, YearBuilt_x_YearRemodAdd, LotArea_x_OverallCond, HalfBath_x_PoolArea, MasVnrArea_x_Fireplaces, LotArea_x_GrLivArea, OverallQual_x_KitchenAbvGr, LotArea_x_TotRmsAbvGrd, OverallQual_x_BedroomAbvGr, LotFrontage_x_LotArea, LotArea_x_OverallQual, YearBuilt_x_KitchenAbvGr, TotalBsmtSF_x_X1stFlrSF, MasVnrArea_x_OpenPorchSF, BsmtUnfSF_x_ScreenPorch, BsmtFinSF1_x_GarageArea, GarageCars_x_YrSold, LotFrontage_x_OverallCond, LotArea_x_Fireplaces, MasVnrArea_x_X2ndFlrSF, OpenPorchSF_x_PoolArea, YearBuilt_x_BedroomAbvGr, MasVnrArea_x_MoSold, LotArea_x_MasVnrArea, TotRmsAbvGrd_x_YrSold, LotArea_x_BsmtFullBath, BsmtUnfSF_x_EnclosedPorch, WoodDeckSF_x_ScreenPorch, GrLivArea_x_FullBath, BsmtFinSF1_x_PoolArea, X1stFlrSF_x_GarageCars, MasVnrArea_x_X3SsnPorch, GarageCars_x_OpenPorchSF, GrLivArea_x_OpenPorchSF, BedroomAbvGr_x_OpenPorchSF, OverallQual_x_TotRmsAbvGrd, LotArea_x_OpenPorchSF, WoodDeckSF_x_OpenPorchSF, BsmtFinSF1_x_BedroomAbvGr, TotalBsmtSF_x_HalfBath, YearBuilt_x_BsmtFinSF1, BsmtFinSF1_x_YrSold, BsmtFinSF1_x_OpenPorchSF, YearRemodAdd_x_GarageCars, Fireplaces_x_OpenPorchSF, FullBath_x_Fireplaces, GrLivArea_x_Fireplaces, HalfBath_x_Fireplaces, BsmtFullBath_x_Fireplaces, OverallCond_x_MasVnrArea, TotalBsmtSF_x_YrSold, YearBuilt_x_TotalBsmtSF, TotRmsAbvGrd_x_OpenPorchSF
interaction_data <- subset(interaction_data, select = c(OverallQual_x_GrLivArea, OverallQual_x_GarageCars, MasVnrArea_x_PoolArea, OverallCond_x_BsmtFinSF1, OverallQual_x_TotalBsmtSF, YearBuilt_x_YearRemodAdd, LotArea_x_OverallCond, HalfBath_x_PoolArea, MasVnrArea_x_Fireplaces, LotArea_x_GrLivArea, OverallQual_x_KitchenAbvGr, LotArea_x_TotRmsAbvGrd, OverallQual_x_BedroomAbvGr, LotFrontage_x_LotArea, LotArea_x_OverallQual, YearBuilt_x_KitchenAbvGr, TotalBsmtSF_x_X1stFlrSF, MasVnrArea_x_OpenPorchSF, BsmtUnfSF_x_ScreenPorch, BsmtFinSF1_x_GarageArea, GarageCars_x_YrSold, LotFrontage_x_OverallCond, LotArea_x_Fireplaces, MasVnrArea_x_X2ndFlrSF, OpenPorchSF_x_PoolArea, YearBuilt_x_BedroomAbvGr, MasVnrArea_x_MoSold, LotArea_x_MasVnrArea, TotRmsAbvGrd_x_YrSold, LotArea_x_BsmtFullBath, BsmtUnfSF_x_EnclosedPorch, WoodDeckSF_x_ScreenPorch, GrLivArea_x_FullBath, BsmtFinSF1_x_PoolArea, X1stFlrSF_x_GarageCars, MasVnrArea_x_X3SsnPorch, GarageCars_x_OpenPorchSF, GrLivArea_x_OpenPorchSF, BedroomAbvGr_x_OpenPorchSF, OverallQual_x_TotRmsAbvGrd, LotArea_x_OpenPorchSF, WoodDeckSF_x_OpenPorchSF, BsmtFinSF1_x_BedroomAbvGr, TotalBsmtSF_x_HalfBath, YearBuilt_x_BsmtFinSF1, BsmtFinSF1_x_YrSold, BsmtFinSF1_x_OpenPorchSF, YearRemodAdd_x_GarageCars, Fireplaces_x_OpenPorchSF, FullBath_x_Fireplaces, GrLivArea_x_Fireplaces, HalfBath_x_Fireplaces, BsmtFullBath_x_Fireplaces, OverallCond_x_MasVnrArea, TotalBsmtSF_x_YrSold, YearBuilt_x_TotalBsmtSF, TotRmsAbvGrd_x_OpenPorchSF))

# LASSOdata <- cbind(numericvaraibles, dcategoryvariables)
# divide into training and test datasets
# Ltrainingset <- LASSOdata[1:1460,]
# Ltestset <- LASSOdata[1461:2919,]

engineered_data <- as.data.frame(cbind(numeric_data, interaction_data, squared_data, factor_data))

##################################
## Numeric Data Transformations ##
##################################

# skewness data profiling
# recode to save results in a data frame instead of printing on the console 
skewness_kurtosis_profiling <- function (dataset) {
  j = 1
  num_row <- sum(sapply(X = dataset, FUN = function(x) is.numeric(x)))
  skewness_kurtosis_info <- as.data.frame(matrix(nrow = num_row, ncol = 2))
  colnames(skewness_kurtosis_info) <- c("Skewness", "Kurtosis")
  # rownames(skewness_kurtosis_info) <- colnames(dataset)
  for (i in 1:ncol(dataset)){
    if (is.numeric(dataset[,i])){
      rownames(skewness_kurtosis_info)[j] <- colnames(dataset[i])
      skewness_kurtosis_info[j,1] <- skewness(x = dataset[,i], na.rm = T)
      skewness_kurtosis_info[j,2] <- kurtosis(x = dataset[,i], na.rm = T)
      # print(vname)
      # print(paste("Skewness", skewness(x = dataset[,i], na.rm = T), sep = " -> "))
      # print(paste("Kurtosis",kurtosis(x = dataset[,i], na.rm = T), sep = " -> "))
      # print("###############")
      # NOTE: prints skewness of the attributes
      # NOTE transformations
      # (1) SQRT()
      # (2) LOG()
      # (3) Inverse / Power
      vname <- colnames(dataset)[i]
      hist(dataset[,i], xlab = vname, main = paste("Histogram of", vname, sep = " "))
      j = j + 1
    }
  }
  return(skewness_kurtosis_info)
}

skewness_kurtosis_info <- skewness_kurtosis_profiling(dataset = engineered_data)
engineered_data_numeric_descriptive_statistics <- numeric_descriptive_statistics(dataset = engineered_data)
engineered_data_factor_descriptive_statistics <- factor_descriptive_statistics(dataset = engineered_data)

skewness_transform <- function (dataset, lsb, usb, trans = c("log", "sqrt", "inverse", "neg_power", "pos_power"), p = NULL) {
  # NOTES:
  # a function that automatically log transforms attributes between a certain rand of skewness
  # dataset: the specified dataset to transform
  # lsb: the lower skew bound
  # usb: the upper skew bound
  # trans: the transformation used e.g. "log", "sqrt", "inverse", "neg_power", "pos_power"
  # p: the power used in conjungtion with trans = "neg_power" or "pos_power"
  # NOTE: inverse transfromation is "neg_power" transformation with p = 1
  # i.e. log transform a variable if it has between [5, 10] skewness
  # IMPORTANT: only really works for non-negative data and makes accomadations for data with 0's
  library(moments)
  for (i in 1:ncol(dataset)){
    if (is.numeric(dataset[,i]) && skewness(dataset[,i], na.rm = T) >= lsb && skewness(dataset[,i], na.rm = T) <= usb) {
      vname <- colnames(dataset)[i]
      print(paste(vname, "had skewness", skewness(dataset[,i], na.rm = T), sep = " "))
      if (trans == "log") {
        # NOTE: cannot take the log of zero or a negative number
        if(0 %in% dataset[,i]) {
          # As 0 is an element of the attribute, add one to attribute as log(0) = -Inf
          # Guarenteed to work for non-negative data
          dataset[,i] <- dataset[,i] + 1
          dataset[,i] <- log(dataset[,i])
        } else {
          # As 0 is not an element of the attibute, straigth transformation
          dataset[,i] <- log(dataset[,i])
        }
      }
      if (trans == "sqrt") {
        # NOTE: cannot take the sqrt of a negative nuumber
        dataset[,i] <- sqrt(dataset[,i])
      }
      if (trans == "neg_power") {
        # NOTE: cannot take the inverse of zero
        if (0 %in% dataset[,i]) {
          # As 0 is an element of the attribute, add one to attribute as 0^-p = Inf
          # Guarenteed to work for non-negative data
          dataset[,i] <- dataset[,i] + 1
          dataset[,i] <- (dataset[,i])^-p
        } else {
          # As 0 is not an element of the attribute, staight transformation
          dataset[,i] <- (dataset[,i])^-p
        }
      }
      if (trans == "pos_power") {
        # NOTE: we can square any number without problems, so straight transformation 
        dataset[,i] <- (dataset[,i])^p
      }
      print(paste(vname, " now has skewness", skewness(dataset[,i], na.rm = T), sep = " "))
    }
  }
  transformed_data <- dataset
  return(transformed_data)
}

transformed_data <- skewness_transform(dataset = imputed_data, lsb = 1, usb = 5, trans = "sqrt")
transformed_data <- skewness_transform(dataset = imputed_data, lsb = 5, usb = 30, trans = "log")
transformed_data <- skewness_transform(dataset = imputed_data, lsb = 30, usb = 45, trans = "neg_power", p = 1)
# transformed_data <- skewness_transform(dataset = imputed_data, lsb = 10, usb = 20, trans = "pos_power", p = 2)
numeric_descriptive_statistics(dataset = transformed_data)

# transform numeric, interaction, squared data
transformed_numeric_data <- skewness_transform(dataset = numeric_data, lsb = 1, usb = 5, trans = "sqrt")
transformed_numeric_data <- skewness_transform(dataset = numeric_data, lsb = 5, usb = 30, trans = "log")
transformed_numeric_data <- skewness_transform(dataset = numeric_data, lsb = 30, usb = 45, trans = "neg_power", p = 1)
transformed_interaction_data <- skewness_transform(dataset = interaction_data, lsb = 1, usb = 5, trans = "sqrt")
transformed_interaction_data <- skewness_transform(dataset = interaction_data, lsb = 5, usb = 30, trans = "log")
transformed_interaction_data <- skewness_transform(dataset = interaction_data, lsb = 30, usb = 45, trans = "neg_power", p = 1)
transformed_squared_data <- skewness_transform(dataset = squared_data, lsb = 1, usb = 5, trans = "sqrt")
transformed_squared_data <- skewness_transform(dataset = squared_data, lsb = 5, usb = 30, trans = "log")
transformed_squared_data <- skewness_transform(dataset = squared_data, lsb = 30, usb = 45, trans = "neg_power", p = 1)

##################################
## Numeric Data Standardisation ##
##################################

# RANGE STANDARDISATION TO [0,1]
range_standardise_data <- function(dataset, lower_bound = 0, upper_bound = 1) {
  standardised_data <- dataset
  # dataset: the specified dataset to range standardised
  # the inner function defines the range standardisation transform, as R does not have one
  # the outer function applies range standardisation to the numeric variables of the specified dataset
  range_normalisation <- function(dataset, lb, ub) {
    # range standardisation [lower bound, upper_bound], e.g. [0, 1]
    # Convert data to have minimum 0 and maximum 1
    mx <- max(dataset)
    mn <- min(dataset)
    ((((dataset - mn) / (mx - mn)) * (ub - lb)) + lb)
  }
  for (i in 1:ncol(dataset)) {
    if (is.numeric(dataset[,i])){
      standardised_data[,i] <- range_normalisation(dataset = dataset[,i], lb = lower_bound, ub = upper_bound)
    }
  }
  return(standardised_data)
}
standardised_data <- range_standardise_data(dataset = transformed_data, lower_bound = 0, upper_bound = 1)
numeric_descriptive_statistics(dataset = standardised_data)
factor_descriptive_statistics(dataset = standardised_data)

# standardise the transformed numeric, interaction and squared data
standardised_numeric_data <- range_standardise_data(dataset = transformed_numeric_data, lower_bound = 0, upper_bound = 1)
standardised_interaction_data <- range_standardise_data(dataset = transformed_interaction_data, lower_bound = 0, upper_bound = 1)
standardised_squared_data <- range_standardise_data(dataset = transformed_squared_data, lower_bound = 0, upper_bound = 1)


# MEAN 0 AND SD 1 STANDARDISTATION
mu_sigma_standardise_data <- function(dataset) {
  # mean 0 and standard deviation 1 standardisation (normalisation)
  # dataset: the dataset to standardise to mean 0 and standard deviation 1
  standardised_data <- dataset
  # scale() is a built in R function() that scales a numeric matrix to mean = 0 and standard deviation = 1
  for (i in 1:ncol(dataset)) {
    if (is.numeric(dataset[,i])){
      standardised_data[,i] <- scale(x = dataset[,i])
    }
  }
  return(standardised_data)
}

standardised_data <- mu_sigma_standardise_data(dataset = transformed_data)
numeric_descriptive_statistics(dataset = standardised_data)
factor_descriptive_statistics(dataset = standardised_data)

# median MAD standardisation
median_MAD_standardise_data <- function(dataset) {
  # convert each variable to have median 0 and median absolute deviation (MAD) of 1
  standardised_data <- dataset
  # dataset: the specified dataset to range standardised
  # the inner function defines the median MAD transform, as R does not have one
  # the outer function applies range standardisation to the numeric variables of the specified dataset
  median_MAD_standardisation <- function(dataset) {
    (dataset - median(dataset, na.rm = T)) / abs(dataset - median(dataset, na.rm = T))
  }
  for (i in 1:ncol(dataset)) {
    if (is.numeric(dataset[,i])){
      standardised_data[,i] <- median_MAD_standardisation(dataset = dataset[,i])
    }
  }
  return(standardised_data)
}
standardised_data <- median_MAD_standardise_data(dataset = transformed_data)
numeric_descriptive_statistics(dataset = standardised_data)
factor_descriptive_statistics(dataset = standardised_data)

##########################################
## Categorical Variables Dummy Encoding ##
##########################################

categorical_dummy_encoding <- function(dataset){
  # we need to dummy encode the categorical variables
  library(dummy)
  # first subset categorical variables
  cvariables <- categories(x = dataset)
  # dummy encode the categorical variables
  dcvariables <- dummy(x = dataset, object = cvariables)
  # we are turning all categorical variables into numeric binary dummy variables
  # as such these numeric binary dummy variables cannot be defined as categorical variables
  # as regression functions cannot handle categorical variables
  # turn all variables into double precision numeric variables
  for (i in 1:length(dcvariables)) {
    dcvariables[,i] <- as.double(dcvariables[,i])
  }
  dcvariables[dcvariables == 1] <- 0
  dcvariables[dcvariables == 2] <- 1
  dcvariables <- as.data.frame(x = dcvariables)
  # subset continuous variables
  # use set operations
  nvariables <- dataset[,setdiff(x = attributes(dataset)$names, y = attributes(cvariables)$names)]
  dummy_encoded_data <- as.data.frame(cbind(nvariables, dcvariables))
  return(dummy_encoded_data)
}

dummy_encoded_data <- categorical_dummy_encoding(dataset = standardised_data)
dummy_encoded_factor_data <- categorical_dummy_encoding(dataset = factor_data)

categorical_dummy_encoding(dataset = factor_data)
categorical_dummy_encoding(dataset = numeric_data)
tail(numeric_descriptive_statistics(dataset = dummy_encoded_data))

########################################################
## Convert Numeric Variables to Categorical Variables ##
########################################################

# poolarea
# change to just pool [yes / no]

######################################################################################################################
## Model Building ####################################################################################################
######################################################################################################################

################################################
## Forward /Backward and Stepwise Elimination ##
################################################

?step()

# I shall perfrom forward / backward and stepwise elimination to distinghish the most important features
# This can be done on both:
# (1) the non-dummy encoded data
# (2) the dummy encoded data

# the dummy encoded data 

# build regression model
# fit full model
SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)
min_model <- lm(SalePrice ~ 1, data = Ltrainingset)
max_model <- formula(lm(SalePrice ~ ., data = Ltrainingset))

# perform forward elimination (note that this takes a very long time)
# forward elmination is the process of building a regression model from the intercept out
# it begins with the intercept only model and adds independant variables accordingly
# it is based on a criterion, R uses the AIC criterion
SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)
min_model <- lm(SalePrice ~ 1, data = Ltrainingset)
max_model <- formula(lm(SalePrice ~ ., data = Ltrainingset))
step(min_model, direction = 'forward', scope = max_model)
fwd_model <- step(min_model, direction = 'forward', scope = max_model)
attributes(fwd_model)
fwd_model$model

# after 40 iterations of forward elmination the best model is:
# SalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood

# backward elimination
SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)
min_model <- lm(SalePrice ~ 1, data = Ltrainingset)
max_model <- formula(lm(SalePrice ~ ., data = Ltrainingset))
bwd_model <- step(object = max_model, scope = min_model, direction = "backward")
attributes(bwd_model)
summary(bwd_model)

# stepwise elimination
SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)
min_model <- lm(SalePrice ~ 1, data = Ltrainingset)
max_model <- formula(lm(SalePrice ~ ., data = Ltrainingset))
stp_model <- step(min_model, direction = 'both', scope = max_model)
summary(stp_model)
predict(stp_model, Ltestset)

# after 41 iterations of forward elmination the best model is:
# SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + Condition2 + SaleCondition + ExterQual + LotArea + YearBuilt + OverallCond + GarageArea + TotalBsmtSF + PoolArea + BedroomAbvGr + Functional + BldgType + MasVnrArea + LandSlope + Condition1 + Foundation + MasVnrType + LowQualFinSF + KitchenAbvGr + Street + SaleType + BsmtFinSF2 + LandContour + ScreenPorch + MoSold + GarageCars + WoodDeckSF + LotConfig + HeatingQC + YearRemodAdd + Fireplaces + RoofStyle + FullBath + Utilities

##############
## Model 40 ##
##############
SalePrice <- train$SalePrice
trainset <- cbind(SalePrice, dummy_encoded_data[1:1460,])
testset <- dummy_encoded_data[1461:2919,]

# NOTE: dummy encoded data used
# Fit Model
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood, data = trainset)
summary(lmodel)
Anova(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# Evaluate Model Fit
# residual vs fits plot
residualPlots(model = lmodel)
# added variable plots with car
avPlots(model = lmodel)

# Process Model
# transform y
#box-cox transformation using car
bct <- boxCox(object = lm(SalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood, data = trainset))
p <- bct$x[which.max(x = bct$y)]
bctSalePrice <- (((trainset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood, data = trainset)

# Re-Evaluate Fit
summary(lmodel)
Anova(lmodel)
residualPlots(model = lmodel)
avPlots(model = lmodel)

# predictions
?predict.lm()
testpred <- predict(object = lmodel, newdata = Ltestset)

##############
## Model 41 ##
##############

# NOTE: non-dummy encoded data used
# Fit Model
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + Condition2 + SaleCondition + ExterQual + LotArea + YearBuilt + OverallCond + GarageArea + TotalBsmtSF + PoolArea + BedroomAbvGr + Functional + BldgType + MasVnrArea + LandSlope + Condition1 + Foundation + MasVnrType + LowQualFinSF + KitchenAbvGr + Street + SaleType + BsmtFinSF2 + LandContour + ScreenPorch + MoSold + GarageCars + WoodDeckSF + LotConfig + HeatingQC + YearRemodAdd + Fireplaces + RoofStyle + FullBath + Utilities, data = trainingset)
summary(lmodel)
Anova(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# Evaluate Model Fit
# residual vs fits plot
residualPlots(model = lmodel)
# added variable plots with car
avPlots(model = lmodel)

# Process Model
# transform y
#box-cox transformation using car
bct <- boxCox(object = lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + Condition2 + SaleCondition + ExterQual + LotArea + YearBuilt + OverallCond + GarageArea + TotalBsmtSF + PoolArea + BedroomAbvGr + Functional + BldgType + MasVnrArea + LandSlope + Condition1 + Foundation + MasVnrType + LowQualFinSF + KitchenAbvGr + Street + SaleType + BsmtFinSF2 + LandContour + ScreenPorch + MoSold + GarageCars + WoodDeckSF + LotConfig + HeatingQC + YearRemodAdd + Fireplaces + RoofStyle + FullBath + Utilities, data = trainingset))
p <- bct$x[which.max(x = bct$y)]
bctSalePrice <- (((trainingset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + Condition2 + SaleCondition + ExterQual + LotArea + YearBuilt + OverallCond + GarageArea + TotalBsmtSF + PoolArea + BedroomAbvGr + Functional + BldgType + MasVnrArea + LandSlope + Condition1 + Foundation + MasVnrType + LowQualFinSF + KitchenAbvGr + Street + SaleType + BsmtFinSF2 + LandContour + ScreenPorch + MoSold + GarageCars + WoodDeckSF + LotConfig + HeatingQC + YearRemodAdd + Fireplaces + RoofStyle + FullBath + Utilities, data = trainingset)

# Re-Evaluate Fit
summary(lmodel)
Anova(lmodel)
residualPlots(model = lmodel)
avPlots(model = lmodel)

# numeric values
numericdata <- cbind(OverallQual, GrLivArea, Neighborhood, KitchenQual, RoofMatl, BsmtFinSF1, MSSubClass, SaleCondition, ExterQual, ExterQual, LotArea, YearBuilt, OverallCond, GarageArea, TotalBsmtSF, PoolArea, BedroomAbvGr, Functional, BldgType, MasVnrArea, LandSlope, Condition1, Foundation, MasVnrType, LowQualFinSF, KitchenAbvGr, Street, SaleType, BsmtFinSF2, LandContour, ScreenPorch, MoSold, GarageCars, WoodDeckSF, LotConfig, HeatingQC, YearRemodAdd, Fireplaces, RoofStyle, FullBath, Utilities)
numericdata2 <- cbind(OverallQual, GrLivArea, Neighborhood, RoofMatl, MSSubClass, SaleCondition, ExterQual, ExterQual, LotArea, YearBuilt, OverallCond, PoolArea, BedroomAbvGr, BldgType, LandSlope, Condition1, Foundation, LowQualFinSF, KitchenAbvGr, Street, LandContour, ScreenPorch, MoSold, WoodDeckSF, LotConfig, HeatingQC, YearRemodAdd, Fireplaces, RoofStyle, FullBath)
summary(numericdata)
summary(numericdata2)
cor(numericdata)
cor(numericdata2)

##############
## Model 77 ##
##############

SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)

# NOTE: dummy encoded data used
# Fit Model
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood + BsmtFinSF2 + YearRemodAdd + Neighborhood_Blmngtn + Exterior2nd_ImStucc + Neighborhood_Mitchel + MSZoning_C..all. + SaleType_Con + MoSold + Foundation_Slab + SaleCondition_AdjLand + Exterior1st_ImStucc + GarageCars + SaleType_CWD + RoofMatl_Membran + LandSlope_Gtl + ExterQual_Fa + Street_Grvl + KitchenAbvGr + BldgType_Duplex + OpenPorchSF + LandContour_Lvl + Neighborhood_NAmes + Neighborhood_Edwards + Neighborhood_OldTown + SaleType_ConLD + Utilities_AllPub + HouseStyle_SLvl + SaleType_Oth + Exterior1st_HdBoard + Exterior1st_Plywood + Neighborhood_NPkVill + Fireplaces + Exterior1st_Stone, data = Ltrainingset)
summary(lmodel)
Anova(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# Evaluate Model Fit
# residual vs fits plot
residualPlots(model = lmodel)
# added variable plots with car
avPlots(model = lmodel)

# Process Model
# transform y
#box-cox transformation using car
bct <- boxCox(object = lm(SalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood + BsmtFinSF2 + YearRemodAdd + Neighborhood_Blmngtn + Exterior2nd_ImStucc + Neighborhood_Mitchel + MSZoning_C..all. + SaleType_Con + MoSold + Foundation_Slab + SaleCondition_AdjLand + Exterior1st_ImStucc + GarageCars + SaleType_CWD + RoofMatl_Membran + LandSlope_Gtl + ExterQual_Fa + Street_Grvl + KitchenAbvGr + BldgType_Duplex + OpenPorchSF + LandContour_Lvl + Neighborhood_NAmes + Neighborhood_Edwards + Neighborhood_OldTown + SaleType_ConLD + Utilities_AllPub + HouseStyle_SLvl + SaleType_Oth + Exterior1st_HdBoard + Exterior1st_Plywood + Neighborhood_NPkVill + Fireplaces + Exterior1st_Stone, data = Ltrainingset))
p <- bct$x[which.max(x = bct$y)]
bctSalePrice <- (((Ltrainingset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood + BsmtFinSF2 + YearRemodAdd + Neighborhood_Blmngtn + Exterior2nd_ImStucc + Neighborhood_Mitchel + MSZoning_C..all. + SaleType_Con + MoSold + Foundation_Slab + SaleCondition_AdjLand + Exterior1st_ImStucc + GarageCars + SaleType_CWD + RoofMatl_Membran + LandSlope_Gtl + ExterQual_Fa + Street_Grvl + KitchenAbvGr + BldgType_Duplex + OpenPorchSF + LandContour_Lvl + Neighborhood_NAmes + Neighborhood_Edwards + Neighborhood_OldTown + SaleType_ConLD + Utilities_AllPub + HouseStyle_SLvl + SaleType_Oth + Exterior1st_HdBoard + Exterior1st_Plywood + Neighborhood_NPkVill + Fireplaces + Exterior1st_Stone, data = Ltrainingset)

# Re-Evaluate Fit
summary(lmodel)
Anova(lmodel)
residualPlots(model = lmodel)
avPlots(model = lmodel)

# predictions
?predict.lm()
testpred <- predict(object = lmodel, newdata = Ltestset)

#####################################################################################################################
## LASSO the trainingset ############################################################################################
#####################################################################################################################

# NOTE: dummy encoded data used
Y <- log(trainingset$SalePrice)
X <- as.matrix(Ltrainingset)

# model selection
LASSOmodel <- glmnet(x = X, y = Y, family = "gaussian", alpha = 1)
LASSOmodel
summary(LASSOmodel)
attributes(LASSOmodel)
LASSOmodel$lambda
# coefficents of LASSO
coef(LASSOmodel)
attributes(coef(LASSOmodel))
# plot coefficients from "glmnet" object
plot(x = LASSOmodel, label = T)
# K-fold cross validation for glmnet
cv.glmnet(x = X, y = Y, nfolds = 10)
# plot cross-validayion curve
# plots the mean-squared error against lambda
plot(cv.glmnet(x = X, y = Y, nfolds = 10))
LASSOmodel$lambda
log(LASSOmodel$lambda)

# model fitting
# chose lambda = 1.145751e-02
LRmodel <- glmnet(x = X, y = Y, alpha = 0, lambda = 1.145751e-02)
coef(LRmodel)
length(coef(LRmodel))
# coefficents
Bhat <- as.matrix(coef(LRmodel)[1:length(coef(LRmodel))])
# fitted values
# Yhat = X%*%Bhat
X <- as.matrix(cbind(rep(1, times = nrow(X)), X))
Yhat <- X%*%Bhat
# residuals
# Ehat = Y - Yhat
ehat <- Y - Yhat
# residuals vs fits plot
plot(x = ehat, y = Yhat, main = "Residuals vs Fits Plot", xlab = "Residuals", ylab = "Fits")

# sum squared error
sum((ehat)^2)
# mean squared error
mean((ehat)^2)
# root mean squared error
sqrt(mean((ehat^2)))
# sum absolute error
sum(abs(ehat))
# mean absolute error
mean(abs(ehat))

# predict new data
# NOTE that s specifies the penalty of the lambda parametre at which predictions are required.
testpred <- predict(object = LASSOmodel, newx = as.matrix(Ltestset), s = 1.145751e-02)

#######################################################################################################################
## Ridge Regression model #############################################################################################
#######################################################################################################################

# NOTE: dummy encoded data used
Y <- log(trainingset$SalePrice)
X <- as.matrix(Ltrainingset)

# model selection
# fit sequence of ridge regression models
RRmodel <- glmnet(x = X, y = Y, alpha = 0)
# query coefficients of ridge regression models
coef(RRmodel)
# plot the coefficients of the ridge regression models accross seperate thresfolds on df
plot(RRmodel, label = T)
# cross-validated mean squared error for the ridge regression models using different levels of lambda
cv.glmnet(x = x, y = y, alpha = 0)
# plot the cross-validated mean squared error for the ridge regression models using different levels of lambda
plot(cv.glmnet(x = X, y = Y, alpha = 0))
RRmodel$lambda
log(RRmodel$lambda)

# model fitting
# chose lambda = 0.36655640
RRmodel <- glmnet(x = X, y = Y, alpha = 0, lambda = 0.36655640)
coef(RRmodel)
# coefficents
Bhat <- as.matrix(coef(RRmodel)[1:length(coef(RRmodel))])
# fitted values
# Yhat = X%*%Bhat
X <- as.matrix(cbind(rep(1, times = nrow(X)), X))
Yhat <- X%*%Bhat
# residuals
# Ehat = Y - Yhat
ehat <- Y - Yhat
# residuals vs fits plot
plot(x = ehat, y = Yhat, main = "Residuals vs Fits Plot", xlab = "Residuals", ylab = "Fits")

# sum squared error
sum((ehat)^2)
# mean squared error
mean((ehat)^2)
# root mean squared error
sqrt(mean((ehat^2)))
# sum absolute error
sum(abs(ehat))
# mean absolute error
mean(abs(ehat))


#############################################################################################################################
## Elastic Net Regression model #############################################################################################
#############################################################################################################################

# NOTE: dummy encoded data used
Y <- log(trainingset$SalePrice)
X <- as.matrix(Ltrainingset)

# model selection
# fit sequence of Elastic Net regression models
ENRmodel <- glmnet(x = X, y = Y, alpha = 0.5)
# query coefficients of ELastic Net regression modelsf
coef(ENRmodel)
# plot the coefficients of the Elastic Net regression models accross seperate thresfolds on df
plot(ENRmodel, label = T)
# cross-validated mean squared error for the Elastic Net regression models using different levels of lambda
cv.glmnet(x = X, y = Y, alpha = 0.5)
# plot the cross-validated mean squared error for the Elastic Net regression models using different levels of lambda
plot(cv.glmnet(x = X, y = Y, alpha = 0.5))
ENRmodel$lambda
log(ENRmodel$lambda)

# model fitting
# chose lambda = 0.0173343744
ENRmodel <- glmnet(x = X, y = Y, alpha = 0, lambda = 0.0173343744)
coef(ENRmodel)
# coefficents
Bhat <- as.matrix(coef(ENRmodel)[1:length(coef(ENRmodel))])
# fitted values
# Yhat = X%*%Bhat
X <- as.matrix(cbind(rep(1, times = nrow(X)), X))
Yhat <- X%*%Bhat
# residuals
# Ehat = Y - Yhat
ehat <- Y - Yhat
# residuals vs fits plot
plot(x = ehat, y = Yhat, main = "Residuals vs Fits Plot", xlab = "Residuals", ylab = "Fits")

# sum squared error
sum((ehat)^2)
# mean squared error
mean((ehat)^2)
# root mean squared error
sqrt(mean((ehat^2)))
# sum absolute error
sum(abs(ehat))
# mean absolute error
mean(abs(ehat))

#######################################################################################################################
## MARS ###############################################################################################################
#######################################################################################################################

# NOTE: dummy encoded data used
Y <- log(train$SalePrice)
X <- as.matrix(Ltrainingset)

# model selection
# fit MARS model
?mars()
marsModel <- mars(x = X, y = Y)
summary(marsModel)
attributes(marsModel)
# note that we can include interactions terms in the mars() command 

# assess fit
# residual vs fits plot
plot(y = marsModel$residuals, x = marsModel$fitted.values, main = "MARS model: Residual vs Fits Plot", ylab = "Residuals", xlab = "Fitted Values")
abline(h = 0)
# sum squared error
sum((marsModel$residuals)^2)
# mean squared error
mean((marsModel$residuals)^2)
# root mean squared error
sqrt(mean((marsModel$residuals^2)))
# sum absolute error
sum(abs(marsModel$residuals))
# mean absolute error
mean(abs(marsModel$residuals))

# make predictions for model
testpred <- predict(object = marsModel, newdata = Ltestset)
testpred <- exp(testpred)

# write to csv file
setwd(dir = "C:/Users/Margaret/Documents/Oisin/R/Kaggle Competitions/House Prices")
Id <- 1461:2919
SalePrice <- testpred
testpreddf <- as.data.frame(cbind(Id, SalePrice))
names(testpreddf) <- c("Id", "SalePrice")
head(testpreddf)
tail(testpreddf)
nrow(testpreddf)
write.table(testpreddf, file = "testpredMARS.csv", sep = ",", quote = F, row.names = F, col.names = T)

######################################################################################################################
## Principle Components Regression (PLS) #############################################################################
######################################################################################################################

SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)

pcrModel <- pcr(formula = log(SalePrice) ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood, ncomp = 5, data = Ltrainingset)
attributes(pcrModel)
summary(pcrModel)

# assess fit
# residual vs fits plot
plot(x = pcrModel$residuals, y = pcrModel$fitted.values, main = "PCR model: Residual vs Fits Plot", xlab = "Residuals", ylab = "Fitted Values")
# sum squared error
sum((pcrModel$residuals)^2)
# mean squared error
mean((pcrModel$residuals)^2)
# root mean squared error
sqrt(mean((pcrModel$residuals^2)))
# sum absolute error
sum(abs(pcrModel$residuals))
# mean absolute error
mean(abs(pcrModel$residuals))

# make predictions for model
testpred <- predict(object = pcrModel, newdata = Ltestset)
testpred <- exp(testpred)

# write to csv file
setwd(dir = "C:/Users/Margaret/Documents/Oisin/R/Kaggle Competitions/House Prices")
Id <- 1461:2919
SalePrice <- testpred
testpreddf <- as.data.frame(cbind(Id, SalePrice))
names(testpreddf) <- c("Id", "SalePrice")
head(testpreddf)
tail(testpreddf)
nrow(testpreddf)
write.table(testpreddf, file = "testpredPCR.csv", sep = ",", quote = F, row.names = F, col.names = T)

######################################################################################################################
## Ensemble Model Bagging OLS ########################################################################################
######################################################################################################################

# NOTE: dummy encoded data used
# model with 40 variables

SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)

# step 1 random observation sampling 
# 1460 observations in our trainingset
# randomly sample 1000 out of 1460 observations for each model
# gonna use model 40
# do via indices

# empty testpred
testpred <- numeric(1459)
for( i in 1:1000){
  # randomly subset the data
  rindices <- sample(x = 1:1460, size = 1000, replace = F)
  randomtrainingdata <- Ltrainingset[rindices, ]
  # fit intial model
  lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood, data = randomtrainingdata)
  # box-cox transformation
  bct <- boxCox(object = lmodel)
  p <- bct$x[which.max(x = bct$y)]
  bctSalePrice <- (((randomtrainingdata$SalePrice)^p) - 1)/(p)
  # fit transformed model
  lmodel1t <- lm(bctSalePrice ~ OverallQual + GrLivArea + BsmtFinSF1 + RoofMatl_ClyTile + KitchenQual_Ex + YearBuilt + MSSubClass + Condition2_PosN + ExterQual_Ex + LotArea + SaleType_New + Neighborhood_NoRidge + OverallCond + BedroomAbvGr + Neighborhood_NridgHt + Neighborhood_StoneBr + RoofMatl_WdShngl + Neighborhood_Crawfor + GarageArea + TotalBsmtSF + Functional_Typ + BldgType_1Fam + PoolArea + SaleCondition_Normal + Condition1_Norm + Neighborhood_BrkSide + HeatingQC_Ex + Exterior1st_BrkFace + Neighborhood_Somerst + LotConfig_CulDSac + WoodDeckSF + MasVnrArea + Neighborhood_NWAmes + Functional_Sev + LowQualFinSF + MasVnrType_BrkFace + ScreenPorch + BldgType_2fmCon + LandContour_HLS + Foundation_Wood, data = randomtrainingdata)
  # make predicitions
  testpred1 <- predict(object = lmodel1t, newdata = Ltestset)
  # back bct tramsform predictions
  testpred1 <- (p*testpred1 + 1)^(1/p)
  # combine predictions
  testpred <- cbind(testpred, testpred1)
}
testpred <- testpred[,-1]
# apply mean row wise
testpred <- apply(testpred, 1, function(x) mean(x))

# esemble model with additional feature selection
# empty testpred
testpred <- numeric(1459)
for( i in 1:100){
  # randomly subset the data
  rindices <- sample(x = 1:1460, size = 1000, replace = F)
  randomtrainingdata <- Ltrainingset[rindices, ]
  # fit intial model
  min_model <- lm(SalePrice ~ 1, data = randomtrainingdata)
  max_model <- formula(lm(SalePrice ~ ., data = randomtrainingdata))
  lmodel <- step(min_model, direction = 'both', scope = max_model)
  # box-cox transformation
  bct <- boxCox(object = lmodel)
  p <- bct$x[which.max(x = bct$y)]
  bctSalePrice <- (((randomtrainingdata$SalePrice)^p) - 1)/(p)
  # fit transformed model
  fmla <- as.character(x = lmodel$call[[2]])
  tfmla <- as.formula(stri_replace_all_regex(str = fmla, pattern = "SalePrice", replacement = "bctSalePrice"))
  tlmodel <- lm(formula = tfmla, data = randomtrainingdata)
  # make predicitions
  testpred1 <- predict(object = tlmodel, newdata = Ltestset)
  # back bct tramsform predictions
  testpred1 <- (p * testpred1 + 1)^(1/p)
  # combine predictions
  testpred <- cbind(testpred, testpred1)
}
testpred <- testpred[,-1]
# apply mean row wise
testpred <- apply(testpred, 1, function(x) mean(x))

SalePrice <- train$SalePrice
Ltrainingset <- cbind(LASSOdata[1:1460,], SalePrice)
# ensemble model with square terms
# empty testpred
testpred <- numeric(1459)
for( i in 1:1000){
  # randomly subset the data
  rindices <- sample(x = 1:1460, size = 1000, replace = F)
  randomtrainingdata <- Ltrainingset[rindices, ]
  # fit intial model
  lmodel <- lm(SalePrice ~ OverallQualSq + GrLivArea + RoofMatl_ClyTile + GarageCarsSq + SaleType_New + Condition2_PosN + TotalBsmtSFSq + OverallQual + KitchenAbvGr + OverallCond + YearBuilt + LotArea + KitchenQual_Ex + Neighborhood_Crawfor + MasVnrAreaSq + BldgType_1Fam + X2ndFlrSFSq + X1stFlrSF + Neighborhood_StoneBr + Neighborhood_NridgHt + Neighborhood_NoRidge + BsmtFinSF1Sq + SaleCondition_Normal + RoofMatl_WdShngl + BsmtFinSF2Sq + Functional_Typ + Exterior1st_BrkFace + Condition1_Norm + PoolAreaSq + ExterQual_Ex + Foundation_PConc + FireplacesSq + Neighborhood_Somerst + LotConfig_CulDSac + Neighborhood_BrkSide + Functional_Maj1 + Functional_Sev + MSZoning_C..all. + GarageAreaSq + LotAreaSq + Heating_OthW + YearRemodAddSq + ScreenPorch + MoSoldSq + PoolArea + Condition1_RRAe + Foundation_Wood + HouseStyle_1.5Fin + LandSlope_Mod + Neighborhood_Mitchel + Street_Grvl + BsmtFullBathSq + Neighborhood_Edwards + LandContour_Low + X3SsnPorch + HeatingQC_Ex + Neighborhood_NPkVill + ExterCond_Gd + WoodDeckSF + WoodDeckSFSq + Exterior1st_MetalSd + Exterior2nd_ImStucc + Exterior1st_ImStucc + TotRmsAbvGrdSq + BedroomAbvGrSq + SaleCondition_AdjLand + X1stFlrSFSq + HouseStyle_2.5Fin + LotShape_IR1 + Neighborhood_NWAmes + FullBathSq + FullBath + BldgType_2fmCon + Neighborhood_OldTown + LandContour_Bnk + Neighborhood_NAmes + RoofStyle_Mansard + SaleType_WD + Utilities_AllPub + OpenPorchSF + OpenPorchSFSq + GrLivAreaSq + EnclosedPorch + RoofMatl_Membran + LandSlope_Gtl + Exterior2nd_Wd.Sdng, data = randomtrainingdata)
  # box-cox transformation
  bct <- boxCox(object = lmodel)
  p <- bct$x[which.max(x = bct$y)]
  bctSalePrice <- (((randomtrainingdata$SalePrice)^p) - 1)/(p)
  # fit transformed model
  lmodel1t <- lm(bctSalePrice ~ OverallQualSq + GrLivArea + RoofMatl_ClyTile + GarageCarsSq + SaleType_New + Condition2_PosN + TotalBsmtSFSq + OverallQual + KitchenAbvGr + OverallCond + YearBuilt + LotArea + KitchenQual_Ex + Neighborhood_Crawfor + MasVnrAreaSq + BldgType_1Fam + X2ndFlrSFSq + X1stFlrSF + Neighborhood_StoneBr + Neighborhood_NridgHt + Neighborhood_NoRidge + BsmtFinSF1Sq + SaleCondition_Normal + RoofMatl_WdShngl + BsmtFinSF2Sq + Functional_Typ + Exterior1st_BrkFace + Condition1_Norm + PoolAreaSq + ExterQual_Ex + Foundation_PConc + FireplacesSq + Neighborhood_Somerst + LotConfig_CulDSac + Neighborhood_BrkSide + Functional_Maj1 + Functional_Sev + MSZoning_C..all. + GarageAreaSq + LotAreaSq + Heating_OthW + YearRemodAddSq + ScreenPorch + MoSoldSq + PoolArea + Condition1_RRAe + Foundation_Wood + HouseStyle_1.5Fin + LandSlope_Mod + Neighborhood_Mitchel + Street_Grvl + BsmtFullBathSq + Neighborhood_Edwards + LandContour_Low + X3SsnPorch + HeatingQC_Ex + Neighborhood_NPkVill + ExterCond_Gd + WoodDeckSF + WoodDeckSFSq + Exterior1st_MetalSd + Exterior2nd_ImStucc + Exterior1st_ImStucc + TotRmsAbvGrdSq + BedroomAbvGrSq + SaleCondition_AdjLand + X1stFlrSFSq + HouseStyle_2.5Fin + LotShape_IR1 + Neighborhood_NWAmes + FullBathSq + FullBath + BldgType_2fmCon + Neighborhood_OldTown + LandContour_Bnk + Neighborhood_NAmes + RoofStyle_Mansard + SaleType_WD + Utilities_AllPub + OpenPorchSF + OpenPorchSFSq + GrLivAreaSq + EnclosedPorch + RoofMatl_Membran + LandSlope_Gtl + Exterior2nd_Wd.Sdng, data = randomtrainingdata)
  # make predicitions
  testpred1 <- predict(object = lmodel1t, newdata = Ltestset)
  # back bct tramsform predictions
  testpred1 <- (p*testpred1 + 1)^(1/p)
  # combine predictions
  testpred <- cbind(testpred, testpred1)
}
testpred <- testpred[,-1]
# apply mean row wise
testpred <- apply(testpred, 1, function(x) mean(x))

################################################################################################################################
## Ensemble Modelling Bagging LASSO ############################################################################################
################################################################################################################################

LASSOdata <- cbind(numericvaraibles, dcategoryvariables)
# divide into training and test datasets
Ltrainingset <- LASSOdata[1:1460,]
Ltestset <- LASSOdata[1461:2919,]

# empty testpred
testpred <- numeric(1459)
for( i in 1:1000) {
  # randomly subset the data
  rindices <- sample(x = 1:1460, size = 1000, replace = F)
  randomtrainingdata <- Ltrainingset[rindices, ]
  SalePrice <- trainingset$SalePrice[rindices]
  # fit LASSO model
  Y <- log(SalePrice)
  X <- as.matrix(randomtrainingdata)
  # predict new data
  testpred1 <- predict.cv.glmnet(object = cv.glmnet(x = X, y = Y, nfolds = 10, alpha = 1), newx = as.matrix(Ltestset))
  # remember to back transform
  testpred1 <- exp(testpred1)
  # combine predictions 
  testpred <- cbind(testpred, testpred1)
}
testpred <- testpred[,-1]
# apply mean row wise
testpred <- apply(testpred, 1, function(x) mean(x))

#######################################################################################################################
## Ensemble Model Bagging MARS model ##################################################################################
#######################################################################################################################

# divide into training and test datasets
Ltrainingset <- LASSOdata[1:1460,]
Ltestset <- LASSOdata[1461:2919,]
# empty testpred
testpred <- numeric(1459)
for( i in 1:100) {
  print(i)
  # randomly subset the data
  # sample roughly 75% of rows and columns
  rindices <- sample(x = 1:1460, size = 1100, replace = F)
  cindices <- sample(x = 1:354, size = 266, replace = F)
  randomtrainingdata <- Ltrainingset[rindices, cindices]
  SalePrice <- train$SalePrice[rindices]
  # fit MARS model
  Y <- log(SalePrice)
  X <- as.matrix(randomtrainingdata)
  marsModel <- mars(x = X, y = Y)
  # predict new data
  # NOTE: need the appropriate columns for the test data
  testpred1 <- predict(object = marsModel, newdata = Ltestset[,cindices])
  # remember to back transform
  testpred1 <- exp(testpred1)
  # combine predictions 
  testpred <- cbind(testpred, testpred1)
}
testpred <- testpred[,-1]
# apply mean row wise
testpred <- apply(testpred, 1, function(x) mean(x))
# write to csv file
setwd(dir = "C:/Users/Margaret/Documents/Oisin/R/Kaggle Competitions/House Prices")
Id <- 1461:2919
SalePrice <- testpred
testpreddf <- as.data.frame(cbind(Id, SalePrice))
names(testpreddf) <- c("Id", "SalePrice")
head(testpreddf)
tail(testpreddf)
nrow(testpreddf)
write.table(testpreddf, file = "testpredMARSEnsemble.csv", sep = ",", quote = F, row.names = F, col.names = T)

# SPLIT INTO TRAINING AND TEST SETS
# training sets
train_standardised_numeric_data <- standardised_numeric_data[1:1460,]
train_dummy_encoded_factor_data <- dummy_encoded_factor_data[1:1460,]
train_standardised_interaction_data <- standardised_interaction_data[1:1460,]
train_standardised_squared_data <- standardised_squared_data[1:1460,]
# testings sets
test_standardised_numeric_data <- standardised_numeric_data[1461:2919,]
test_dummy_encoded_factor_data <- dummy_encoded_factor_data[1461:2919,]
test_standardised_interaction_data <- standardised_interaction_data[1461:2919,]
test_standardised_squared_data <- standardised_squared_data[1461:2919,]
# want to always use the train_standardised_numeric_data
# but in addition sample from:
# (1) train_dummy_encoded_factor_data
# (2) train_standardised_interaction_data
# (3) train_standardised_squared_data
# empty testpred
testpred <- numeric(1459)
for( i in 1:100) {
  print(i)
  # (1) randomly sample 80% of the rows of the train_standardised_numeric_data
  rindices <- sample(x = 1:nrow(train_standardised_numeric_data), size = round(nrow(train_standardised_numeric_data) * 0.8), replace = F)
  random_numeric_data <- train_standardised_numeric_data[rindices, ]
  # (2) randomly 80% of the train_dummy_encoded_factor_data columns
  cindicesd <- sample(x = 1:ncol(train_dummy_encoded_factor_data), size = round(ncol(train_dummy_encoded_factor_data) * 0.8), replace = F)
  random_dummy_encoded_factor_data <- train_dummy_encoded_factor_data[rindices, cindicesd]
  # (3) randomly sample 80% of the train_standardised_interaction_data columns
  cindicesi <- sample(x = 1:ncol(train_standardised_interaction_data), size = round(ncol(train_standardised_interaction_data) * 0.8), replace = F)
  random_interaction_data <- train_standardised_interaction_data[rindices, cindicesi]
  # (4) randomly sample 80% of the train_standardised_squared_data columns
  cindicess <- sample(x = 1:ncol(train_standardised_squared_data), size = round(ncol(train_standardised_squared_data) * 0.8), replace = F)
  random_squared_data <- train_standardised_squared_data[rindices, cindicess]
  # combine the randomly sampled data
  X <- as.matrix(cbind(random_numeric_data, random_dummy_encoded_factor_data, random_interaction_data, random_squared_data))
  # define the reponse variable from train dataset
  Y <- log(train$SalePrice[rindices])
  # fit MARS model
  marsModel <- mars(x = X, y = Y)
  # predict new data
  # NOTE: need the appropriate columns for the test data
  testset <- cbind(test_standardised_numeric_data, test_dummy_encoded_factor_data[,cindicesd], test_standardised_interaction_data[,cindicesi], test_standardised_squared_data[,cindicess])
  testpred1 <- predict(object = marsModel, newdata = testset)
  # remember to back transform
  testpred1 <- exp(testpred1)
  # combine predictions 
  testpred <- cbind(testpred, testpred1)
}
testpred <- testpred[,-1]
# apply mean row wise
testpred <- apply(testpred, 1, function(x) mean(x))
# write to csv file
setwd(dir = "C:/Users/Margaret/Documents/Oisin/R/Kaggle Competitions/House Prices")
Id <- 1461:2919
SalePrice <- testpred
testpreddf <- as.data.frame(cbind(Id, SalePrice))
names(testpreddf) <- c("Id", "SalePrice")
head(testpreddf)
tail(testpreddf)
nrow(testpreddf)
write.table(testpreddf, file = "testpredMARSEnsemble.csv", sep = ",", quote = F, row.names = F, col.names = T)

######################################################################################################################
## Neural Networks ###################################################################################################
######################################################################################################################

SalePrice <- train$SalePrice
trainset <- dummy_encoded_data[1:1460,]
testset <- dummy_encoded_data[1461:2919,]
library(nnet)
X <- trainset
# fit nueral network model
nnModel <- nnet(x = X, y = SalePrice, size = 1, decay = 0.1, maxit = 100, rang = 0.7)
nnModel <- nnet(x = X, y = SalePrice, size = 2, decay = 0.1, maxit = 100, rang = 0.7)
nnModel <- nnet(x = X, y = SalePrice, size = 3, decay = 0.1, maxit = 100, rang = 0.7)
# assess fit
# residual vs fits plot
plot(y = nnModel$residuals, x = nnModel$fitted.values, main = "Neural Network model: Residual vs Fits Plot", ylab = "Residuals", xlab = "Fitted Values")
abline(h = 0)
# sum squared error
sum((nnModel$residuals)^2)
# mean squared error
mean((nnModel$residuals)^2)
# root mean squared error
sqrt(mean((nnModel$residuals^2)))
# sum absolute error
sum(abs(nnModel$residuals))
# mean absolute error
mean(abs(nnModel$residuals))


######################################################################################################################
## Support Vector MAchines Regression ################################################################################
######################################################################################################################

SalePrice <- train$SalePrice
trainset <- dummy_encoded_data[1:1460,]
testset <- dummy_encoded_data[1461:2919,]
library(kernlab)
X <- as.matrix(trainset)
svrModel <- ksvm(x = X, y = log(SalePrice), scaled = F, type = "eps-svr", kernel = "rbfdot")
svrModel <- ksvm(x = X, y = log(SalePrice), scaled = F, type = "nu-svr", kernel = "rbfdot")
testpred <- predict(object = svrModel, newdata = testset)



###########################################################################################################################
## Predict from the model #################################################################################################
###########################################################################################################################

# predictions
?predict.lm()
testpred <- predict(object = lmodel, newdata = testset)
# backwards transformsations
# remember log inverse transformation
testpred <- exp(testpred)
# remember bct inverse transformation
testpred <- (p*testpred + 1)^(1/p)
head(testpred)

######################################################################################################################
## Write to CSV file #################################################################################################
######################################################################################################################

setwd(dir = "C:/Users/Margaret/Documents/Oisin/R/Kaggle Competitions/House Prices")
Id <- 1461:2919
SalePrice <- testpred
testpreddf <- as.data.frame(cbind(Id, SalePrice))
names(testpreddf) <- c("Id", "SalePrice")
head(testpreddf)
tail(testpreddf)
nrow(testpreddf)
write.table(testpreddf, file = "testpredOLSensemble.csv", sep = ",", quote = F, row.names = F, col.names = T)

#################################################################################################################
## Appendix #####################################################################################################
#################################################################################################################


# impute for missing values

# MSZoning (Categorical)
summary(reduceddata)
reduceddata$MSZoning <- as.factor(MSZoning)
summary(reduceddata$MSZoning)
which(is.na(reduceddata$MSZoning)) 
levele
reduceddata$MSZoning[which(is.na(reduceddata$MSZoning))] <- 'RL'

# Utilities (Categorical)
reduceddata$Utilities <- as.factor(x = reduceddata$Utilities)
summary(reduceddata$Utilities)
reduceddata$Utilities[which(is.na(reduceddata$Utilities))] <- 'AllPub'

#Exterior1st (Categorical)
reduceddata$Exterior1st <- as.factor(x = reduceddata$Exterior1st)
summary(reduceddata$Exterior1st)
reduceddata$Exterior1st[which(is.na(reduceddata$Exterior1st))] <- 'VinylSd'

#Exterior2nd (Categorical)
reduceddata$Exterior2nd <- as.factor(x = reduceddata$Exterior2nd)
summary(reduceddata$Exterior2nd)
reduceddata$Exterior2nd[which(is.na(reduceddata$Exterior2nd))] <- 'VinylSd'

#MasVnrType (Categorical)
reduceddata$MasVnrType <- as.factor(x = reduceddata$MasVnrType)
summary(reduceddata$MasVnrType)
reduceddata$MasVnrType[which(is.na(reduceddata$MasVnrType))] <- 'None'

#MasVnrArea
summary(reduceddata$MasVnrArea)
reduceddata$MasVnrArea[which(is.na(reduceddata$MasVnrArea))] <- mean(reduceddata$MasVnrArea, na.rm = T)
# reduceddata$MasVnrArea[which(is.na(reduceddata$MasVnrArea))] <- '0'

#BsmtFinSF1 (Numeric)
summary(reduceddata$BsmtFinSF1) 
reduceddata$BsmtFinSF1[which(is.na(reduceddata$BsmtFinSF1))] <- mean(reduceddata$BsmtFinSF1, na.rm = T)

#BsmtFinSF2 (Numeric)
summary(reduceddata$BsmtFinSF2) 
reduceddata$BsmtFinSF2[which(is.na(reduceddata$BsmtFinSF2))] <- mean(reduceddata$BsmtFinSF2, na.rm = T)

#BsmtUnfSF (Numeric)
summary(reduceddata$BsmtUnfSF) 
reduceddata$BsmtUnfSF[which(is.na(reduceddata$BsmtUnfSF))] <- mean(reduceddata$BsmtUnfSF, na.rm = T)

#TotalBsmtSF (Numeric)
summary(reduceddata$TotalBsmtSF) 
reduceddata$TotalBsmtSF[which(is.na(reduceddata$TotalBsmtSF))] <- mean(reduceddata$TotalBsmtSF, na.rm = T)

#Electrical (Categorical)
reduceddata$Electrical <- as.factor(x = reduceddata$Electrical)
summary(reduceddata$Electrical) 
reduceddata$Electrical[which(is.na(reduceddata$Electrical))] <- 'SBrkr'

#BsmtFullBath (Numeric)
summary(reduceddata$BsmtFullBath) 
reduceddata$BsmtFullBath[which(is.na(reduceddata$BsmtFullBath))] <- mean(reduceddata$BsmtFullBath, na.rm = T)

#BsmtHalfBath (Numeric)
summary(reduceddata$BsmtHalfBath) 
reduceddata$BsmtHalfBath[which(is.na(reduceddata$BsmtHalfBath))] <- mean(reduceddata$BsmtHalfBath, na.rm = T)

#KitchenQual (Categorical)
reduceddata$KitchenQual <- as.factor(x = reduceddata$KitchenQual)
summary(reduceddata$KitchenQual) 
reduceddata$KitchenQual[which(is.na(reduceddata$KitchenQual))] <- 'TA'

#Functional (Categorical)
reduceddata$Functional <- as.factor(x = reduceddata$Functional)
summary(reduceddata$Functional) 
reduceddata$Functional[which(is.na(reduceddata$Functional))] <- 'Typ'

#GarageCars
summary(reduceddata$GarageCars) 
reduceddata$GarageCars[which(is.na(reduceddata$GarageCars))] <- mean(reduceddata$GarageCars, na.rm = T)

#GarageArea
summary(reduceddata$GarageArea) 
reduceddata$GarageArea[which(is.na(reduceddata$GarageArea))] <- mean(reduceddata$GarageArea, na.rm = T)

#SaleType
reduceddata$SaleType <- as.factor(x = reduceddata$SaleType)
summary(reduceddata$SaleType) 
reduceddata$SaleType[which(is.na(reduceddata$SaleType))] <- 'WD'

sapply(reduceddata, function(x) sum(is.na(x)))
summary(reduceddata)

##########################################
## Dummy Encoding Categorical Variables ##
##########################################

Ltrainingset <- reduceddata
# we need to dummy encode the categorical variables
# subset categorical variables
cvariables <- categories(Ltrainingset)
cvariables
attributes(cvariables)$names
dcategoryvariables <- dummy(x = Ltrainingset, object = categories(x = Ltrainingset))
# turn all variables into double precision numeric variables
for (i in 1:length(dcategoryvariables)) {
  dcategoryvariables[,i] <- as.double(dcategoryvariables[,i])
}
dcategoryvariables[dcategoryvariables == 1] <- 0
dcategoryvariables[dcategoryvariables == 2] <- 1
dcategoryvariables <- as.data.frame(x = dcategoryvariables)
# subset continuous variables
# use set operations
nvariables <- setdiff(x = attributes(Ltrainingset)$names, y = attributes(cvariables)$names)
nvariables <- setdiff(x = nvariables, y = "SalePrice")
numericvaraibles <- Ltrainingset[nvariables]
# data profiling
# for (i in 1:length(numericvaraibles)){
#   xaxis <- attributes(numericvaraibles)$names[i]
#   hist(numericvaraibles[,i], xlab = xaxis, main = paste("Histogram for ", xaxis))
# }
#
# transform skewed numeric variables
# for (i in 1:length(numericvaraibles)){
#   p_value <- agostino.test(numericvaraibles[,i], alternative = "two.sided")[attributes(agostino.test(numericvaraibles[,i], alternative = "two.sided"))$names[2]][1][[1]]
#   ifelse(test = p_value < 0.05, numericvaraibles[ ,i] <- log(numericvaraibles[ ,i] + 1), numericvaraibles[ ,i] <- numericvaraibles[ ,i])
# }

# turn all variables into double precision numeric variables
for (i in 1:length(numericvaraibles)) {
  numericvaraibles[,i] <- as.double(numericvaraibles[,i])
}
LASSOdata <- cbind(numericvaraibles, dcategoryvariables)
# divide into training and test datasets
Ltrainingset <- LASSOdata[1:1460,]
Ltestset <- LASSOdata[1461:2919,]

###############
## AIC Model ##
###############

# build model with 10 independent variables
AICmodel <- lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + SaleCondition + ExterQual, data = trainingset)
summary(AICmodel)
hist(AICmodel$residuals)
# normality test
?shapiro.test
shapiro.test(AICmodel$residuals)
# Ho: the sample comes from a normal distribution
# Ha: the sample does not come from a normal distribution

###########
## Plots ##
###########

# Histogram of residuals
hist(AICmodel$residuals)
# residauls vs fits plot
plot(AICmodel$fitted.values, AICmodel$residuals)
# major non-constant variance

# residual vs predictors
plot(trainingset$OverallQual, AICmodel$residuals)
plot(trainingset$GrLivArea, AICmodel$residuals)
plot(trainingset$Neighborhood, AICmodel$residuals)
plot(trainingset$KitchenQual, AICmodel$residuals)
plot(trainingset$RoofMatl, AICmodel$residuals)
plot(trainingset$BsmtFinSF1, AICmodel$residuals)
plot(trainingset$MSSubClass, AICmodel$residuals)
plot(trainingset$Condition2, AICmodel$residuals)
plot(trainingset$SaleCondition, AICmodel$residuals)
plot(trainingset$ExterQual, AICmodel$residuals)

# fits vs predictor (yhat vs y)
plot(trainingset$SalePrice, AICmodel$fitted.values)

# Variance Stabilisation
library(acepack)
avas(cbind(OverallQual, GarageArea), SalePrice)

############################
## Box-Cox transformation ##
############################

#response y must be positive
AICmodel <- lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + Condition2 + SaleCondition + ExterQual, data = trainingset)
bct <- boxcox(object = AICmodel)
which.max(x = bct$y)
p <- bct$x[58]
#optimal lambda = 57
#as lambda != 0 our transformed resonse variable is
bctSalePrice <- (((trainingset$SalePrice)^p) - 1)/(p)
bctAICmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + Condition2 + SaleCondition + ExterQual, data = trainingset)

summary(bctAICmodel)
hist(bctAICmodel$residuals)
shapiro.test(bctAICmodel$residuals)
# residauls vs fits plot
plot(AICmodel$fitted.values, bctAICmodel$residuals)
# major non-constant variance

# residual vs predictors
plot(trainingset$OverallQual, bctAICmodel$residuals)
plot(trainingset$GrLivArea, bctAICmodel$residuals)
plot(trainingset$Neighborhood, bctAICmodel$residuals)
plot(trainingset$KitchenQual, bctAICmodel$residuals)
plot(trainingset$RoofMatl, bctAICmodel$residuals)
plot(trainingset$BsmtFinSF1, bctAICmodel$residuals)
plot(trainingset$MSSubClass, bctAICmodel$residuals)
plot(trainingset$Condition2, bctAICmodel$residuals)
plot(trainingset$SaleCondition, bctAICmodel$residuals)
plot(trainingset$ExterQual, bctAICmodel$residuals)

# fits vs predictor (yhat vs y)
plot(bctSalePrice, bctAICmodel$fitted.values)

############################
## Predict from the model ##
############################

# update test set 
testset <- subset(testset, select = c(OverallQual, GarageArea, TotalBsmtSF, BsmtUnfSF, Neighborhood, MasVnrArea, ExterCond, Exterior2nd, Exterior1st))

# predictions
?predict.lm()
predictions <- predict(object = lmodel, newdata = testset)

################################################
## Forward /Backward and Stepwise Elimination ##
################################################

?step()

# I shall perfrom forward / backward and stepwise elimination to distinghish the most important features
# This can be done on both:
# (1) the non-dummy encoded data
# (2) the dummy encoded data

####################################
## (1) the non-dummy encoded data ##
####################################

# build regression model
# fit full model
min_model <- lm(SalePrice ~ 1, data = trainingset)
max_model <- formula(lm(SalePrice ~ ., data = trainingset))

# forward elimination
# perform forward elimination (note that this takes a very long time)
# forward elmination is the process of building a regression model from the intercept out
# it begins with the intercept only model and adds independant variables accordingly
# it is based on a criterion, R uses the AIC criterion
fwd_model <- step(min_model, direction = 'forward', scope = max_model)
# after 41 iterations of forward elmination the best model is:
# SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + Condition2 + SaleCondition + ExterQual + LotArea + YearBuilt + OverallCond + GarageArea + TotalBsmtSF + PoolArea + BedroomAbvGr + Functional + BldgType + MasVnrArea + LandSlope + Condition1 + Foundation + MasVnrType + LowQualFinSF + KitchenAbvGr + Street + SaleType + BsmtFinSF2 + LandContour + ScreenPorch + MoSold + GarageCars + WoodDeckSF + LotConfig + HeatingQC + YearRemodAdd + Fireplaces + RoofStyle + FullBath + Utilities
summary(fwd_model)

# backward elimination
min_model<- formula(lm(SalePrice ~ 1, data = trainingset))
max_model <- lm(SalePrice ~ ., data = trainingset)
bwd_model <- step(object = max_model, scope = min_model, direction = "backward")
attributes(bwd_model)
summary(bwd_model)

# stepwise elimination
min_model <- lm(SalePrice ~ 1, data = trainingset)
max_model <- formula(lm(SalePrice ~ ., data = trainingset))
stp_model <- step(min_model, direction = 'both', scope = max_model)
summary(stp_model)

#############
## model 1 ##
#############

# NOTE: non-dummy encoded data used
# fit model
lmodel <- lm(SalePrice ~ OverallQual, data = trainingset)
summary(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# evaluate model fit
# overall model
plot(x = trainingset$OverallQual, y = trainingset$SalePrice)
cor(x = trainingset$OverallQual, y = trainingset$SalePrice)
# residual vs fits plot
plot(x = lmodel$fitted.values, y = lmodel$residuals)
# linearity violated, non-constant variance
# residual vs predictor plot
plot(x = trainingset$OverallQual, y = lmodel$residuals)
# histogram of SalePrice
hist(trainingset$SalePrice)
hist(log(trainingset$SalePrice))
# normatlity tests
shapiro.test(x = log(trainingset$SalePrice))
ad.test(x = log(trainingset$SalePrice))
cvm.test(x = log(trainingset$SalePrice))
lillie.test(x = log(trainingset$SalePrice))
sf.test(x = log(trainingset$SalePrice))

# Process Model
# transform y
#box-cox transformation
lmodel <- lm(SalePrice ~ OverallQual, data = trainingset)
bct <- boxcox(object = lmodel)
which.max(x = bct$y)
p <- bct$x[52]
#optimal lambda = 52
#as lambda != 0 our transformed resonse variable is
bctSalePrice <- (((trainingset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual, data = trainingset)
# log transformation
lmodel <- lm(log(SalePrice) ~ OverallQual, data = trainingset)
# variance stabilisation transformation
vst <- avas(x = trainingset$OverallQual, y = trainingset$SalePrice)
summary(vst)
attributes(vst)
lmodel <- lm(vst$ty ~ vst$tx, data = trainingset)

# Re-fit model
summary(lmodel)
hist(lmodel$residuals)
# normality tests
shapiro.test(x = lmodel$residuals)
ad.test(x = lmodel$residuals)
cvm.test(x = lmodel$residuals)
lillie.test(x = lmodel$residuals)
sf.test(x = lmodel$residuals)

# Re-Evaluate Fit
plot(x = lmodel$fitted.values, y = lmodel$residuals)
# linearity violated, non-constant variance
plot(x = trainingset$OverallQual, y = lmodel$residuals)

#############
## Model 2 ##
#############

# NOTE: non-dummy encoded data used
# Fit Model
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea, data = trainingset)
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + OverallQual*GrLivArea, data = trainingset)
lmodel <- lm(SalePrice ~ OverallQual + I(OverallQual^2) + GrLivArea + I(GrLivArea^2), data = trainingset)
summary(lmodel)
Anova(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# Evaluate Model Fit
# histogram of SalePrice
hist(trainingset$SalePrice)
hist(log(trainingset$SalePrice))
# normatlity tests
shapiro.test(x = log(trainingset$SalePrice))
ad.test(x = log(trainingset$SalePrice))
cvm.test(x = log(trainingset$SalePrice))
lillie.test(x = log(trainingset$SalePrice))
sf.test(x = log(trainingset$SalePrice))
# overall model
persp(x = trainingset$OverallQual, y = trainingset$SalePrice, z = trainingset$GrLivArea)
pairs(cbind(trainingset$OverallQual, trainingset$GrLivArea, trainingset$SalePrice), labels = c("OverallQual", "GrLivArea", "SalePrice"))
cor(x = trainingset$OverallQual, y = trainingset$SalePrice)
cor(x = trainingset$GrLivArea, y = trainingset$SalePrice)
cor(x = trainingset$GrLivArea, y = trainingset$OverallQual)
# residual vs fits plot
plot(x = lmodel$fitted.values, y = lmodel$residuals)
residualPlots(model = lmodel)
# linearity violated, non-constant variance
# residual vs predictor plot
plot(x = trainingset$OverallQual, y = lmodel$residuals)
plot(x = trainingset$GrLivArea, y = lmodel$residuals)
# added variable plot for OverallQual
yaxis <- lm(SalePrice ~ GrLivArea, data = trainingset)$residuals
xaxis <- lm(OverallQual ~ GrLivArea, data = trainingset)$residuals
plot(x = lm(OverallQual ~ GrLivArea, data = trainingset)$residuals, y = lm(SalePrice ~ GrLivArea, data = trainingset)$residuals)
# OverallQual is useful as points are tightly clustered around a line
# added variable plot for GrLivArea
yaxis <- lm(SalePrice ~ OverallQual, data = trainingset)$residuals
xaxis <- lm(GrLivArea ~ OverallQual, data = trainingset)$residuals
plot(x = lm(GrLivArea ~ OverallQual, data = trainingset)$residuals, y = lm(SalePrice ~ OverallQual, data = trainingset)$residuals)
# added variable plots with car
avPlots(model = lmodel)
# outliers in GrLiveArea
head(sort(trainingset$GrLivArea, decreasing = T))
summary(trainingset$GrLivArea)
trainingset[which(trainingset$GrLivArea > 4000),]

# Process Model
# transform y
#box-cox transformation using car
bct <- boxCox(object = lm(SalePrice ~ OverallQual + I(OverallQual^2) + GrLivArea + I(GrLivArea^2), data = trainingset))
p <- bct$x[which.max(x = bct$y)]
bctSalePrice <- (((trainingset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual + I(OverallQual^2) + GrLivArea + I(GrLivArea^2), data = trainingset)
# log transformation
lmodel <- lm(log(SalePrice) ~ OverallQual + GrLivArea, data = trainingset)
# variance stabilisation transformation
vst <- avas(x = trainingset$OverallQual, y = trainingset$SalePrice)
summary(vst)
attributes(vst)
lmodel <- lm(vst$ty ~ vst$tx, data = trainingset)
# transform x
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea, data = trainingset)
summary(trainingset$GrLivArea)
hist(GrLivArea)

# Re-Evaluate Fit
summary(lmodel)
Anova(lmodel)
residualPlots(model = lmodel)
avPlots(model = lmodel)
hist(lmodel$residuals)

#############
## Model 3 ##
#############

# NOTE: non-dummy encoded data used
# Fit Model
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood, data = trainingset)
summary(lmodel)
Anova(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# Evaluate Model Fit
# histogram of SalePrice
hist(trainingset$SalePrice)
hist(log(trainingset$SalePrice))
# normatlity tests
shapiro.test(x = log(trainingset$SalePrice))
ad.test(x = log(trainingset$SalePrice))
cvm.test(x = log(trainingset$SalePrice))
lillie.test(x = log(trainingset$SalePrice))
sf.test(x = log(trainingset$SalePrice))
# overall model
pairs(cbind(trainingset$OverallQual, trainingset$GrLivArea, trainingset$Neighborhood, trainingset$SalePrice), labels = c("OverallQual", "GrLivArea", "Neighbour", "SalePrice"))
cor(x = trainingset$OverallQual, y = trainingset$SalePrice)
cor(x = trainingset$GrLivArea, y = trainingset$SalePrice)
cor(x = trainingset$GrLivArea, y = trainingset$OverallQual)
# residual plots
residualPlots(model = lmodel)
# added variable plots with car
avPlots(model = lmodel)

# Process Model
# transform y
#box-cox transformation using car
bct <- boxCox(object = lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood, data = trainingset))
p <- bct$x[which.max(x = bct$y)]
bctSalePrice <- (((trainingset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea + Neighborhood, data = trainingset)
# log transformation
lmodel <- lm(log(SalePrice) ~ OverallQual + GrLivArea, data = trainingset)
# variance stabilisation transformation
vst <- avas(x = trainingset$OverallQual, y = trainingset$SalePrice)
summary(vst)
attributes(vst)
lmodel <- lm(vst$ty ~ vst$tx, data = trainingset)
# transform x
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea, data = trainingset)
summary(trainingset$GrLivArea)
hist(GrLivArea)

# Re-Evaluate Fit
summary(lmodel)
Anova(lmodel)
residualPlots(model = lmodel)
avPlots(model = lmodel)
hist(lmodel$residuals)

#############
## Model 4 ##
#############

# NOTE: non-dummy encoded data used
# Fit Model
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual, data = trainingset)
summary(lmodel)
Anova(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# Evaluate Model Fit
# overall model
pairs(cbind(trainingset$OverallQual, trainingset$GrLivArea, trainingset$KitchenQual, trainingset$SalePrice), labels = c("OverallQual", "GrLivArea", "KitchenQual", "SalePrice"))
# residual vs fits plot
residualPlots(model = lmodel)
# added variable plots with car
avPlots(model = lmodel)

# Process Model
# transform y
#box-cox transformation using car
bct <- boxCox(object = lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual, data = trainingset))
p <- bct$x[which.max(x = bct$y)]
bctSalePrice <- (((trainingset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual, data = trainingset)

# Re-Evaluate Fit
summary(lmodel)
Anova(lmodel)
residualPlots(model = lmodel)
avPlots(model = lmodel)

##############
## Model 10 ##
##############

# NOTE: non-dummy encoded data used
# Fit Model
lmodel <- lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + SaleCondition + ExterQual, data = trainingset)
summary(lmodel)
Anova(lmodel)
hist(lmodel$residuals)
shapiro.test(x = lmodel$residuals)

# Evaluate Model Fit
# residual vs fits plot
residualPlots(model = lmodel)
# added variable plots with car
avPlots(model = lmodel)

# Process Model
# transform y
#box-cox transformation using car
bct <- boxCox(object = lm(SalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + SaleCondition + ExterQual, data = trainingset))
p <- bct$x[which.max(x = bct$y)]
bctSalePrice <- (((trainingset$SalePrice)^p) - 1)/(p)
lmodel <- lm(bctSalePrice ~ OverallQual + GrLivArea + Neighborhood + KitchenQual + RoofMatl + BsmtFinSF1 + MSSubClass + SaleCondition + ExterQual, data = trainingset)

# Re-Evaluate Fit
summary(lmodel)
Anova(lmodel)
residualPlots(model = lmodel)
avPlots(model = lmodel)


#########################
## MARS Ensemble Model ##
#########################

MARS_Ensemble_Model_Predictions <- function (num_models, num_rows, num_cols, train_data, response_variable, test_data) {
  # MODEL DETAILS
  # num_models: The number of MARS models to be generated
  # num_rows: the number of rows to sample when bagging
  # num_cols: the number of cols to sample when bagging
  # train_data: the training data to be used (excludes response variable)
  # response_variable: the response variable to be used
  # test_data: the test data to predict
  # empty testpred
  testpred <- numeric(length = nrow(test_data))
  for(i in 1:num_models) {
    print(i)
    # TRAINING DATA
    # randomly subset the data
    # sample roughly 75% of rows and columns
    rindices <- sample(x = 1:nrow(train_data), size = num_rows, replace = F)
    cindices <- sample(x = 1:ncol(train_data), size = num_cols, replace = F)
    randomtrainingdata <- train_data[rindices, cindices]
    # fit MARS model
    Y <- t(as.vector(response_variable))
    X <- as.matrix(randomtrainingdata)
    marsModel <- mars(x = X, y = Y, prune = T, forward.step = T)
    # PREDICT TEST DATA
    # NOTE: need the appropriate columns for the test data
    testpred1 <- predict(object = marsModel, newdata = test_data[ ,cindices])
    # remember to back transform
    testpred1 <- exp(testpred1)
    # combine predictions 
    testpred <- cbind(testpred, testpred1)
  }
  testpred <- testpred[,-1]
  # apply mean row wise
  testpred <- apply(X = testpred, MARGIN = 1, FUN = function(x) mean(x))
  return(testpred)
}

testpred <- MARS_Ensemble_Model_Predictions(num_models = 2, num_rows = 1100, num_cols = 200, train_data = Ltrainingset, response_variable = train$SalePrice, test_data = Ltestset)
