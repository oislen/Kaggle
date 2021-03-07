# load relevant libraries
library(earth)
library(tidyverse)
library(ineq)

# set data directories
data_dir= 'C:/Users/User/Documents/GitHub/Kaggle/competitions/HousePrices/data'

# set file names
engin_fname = 'engin.csv'
preds_mars_fname = 'preds_mars.csv'

# create full file paths
engin_data_fpath = file.path(data_dir, engin_fname)
pred_data_fpath = file.path(data_dir, preds_mars_fname)

#################
#-- Load Data --#
#################

# load in base data
base_data = read.csv(file = engin_data_fpath)

# split out the datasets
train = base_data[base_data$Dataset == 'train',]
test = base_data[base_data$Dataset == 'test',]

# split up columns
reference = c('Id', 'Dataset')
target = 'logSalePrice'
model_cols = colnames(base_data)
predictors = model_cols[!model_cols %in% c(reference, target)]

###################
#-- Train Model --#
###################

print('Training Model ...')

# duplicate training data
df_train = train

# create exog and endog to train model
X = df_train[,predictors]
y = df_train[target]

# fit mars model
mars = earth(x = X,
             y = y,
             trace = 2,
             glm = list(family = gaussian),
             degree = 1,
             thresh = 0.001,
             #penalty = 2,
             nk = 30
)

# feature importance
evimp(mars)
summary(mars)
plot(mars)

###################
#-- Predictions --#
###################

#-- Valid and Test Predicitons --#

print('Outputting training results for evaluation ...')

# create exog, endog and predictions for test set
exog_test = test[predictors]

# make predictions and back transform predictions with exponent to get predict SalePrice
pred_test = expm1(predict(mars, newdata = exog_test, type = "response"))

# calculate gini
ineq(x = pred_test, type = 'Gini')

# write into test file
test['SalePrice'] = pred_test

# extract test file for outputting
test_output = test[,c('Id', 'SalePrice')]

# write output test file as .csv file
write.csv(x = test_output, 
          file = pred_data_fpath, 
          row.names = FALSE)
