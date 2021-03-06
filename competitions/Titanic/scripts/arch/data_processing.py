# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:26:42 2018

@author: oisin_000
"""

"""
###############################################################################
## Preliminaries ##############################################################
###############################################################################
"""

#-- load libraries --#

print('loading libraries')

# os will be used for set working directory
import os

# pandas will be used for data manipulation
import pandas as pd

# numpy will be used for functions
import numpy as np

# sklearn will be used for the modelling and classification
from sklearn import feature_selection
from sklearn import preprocessing

#-- set working directory --###################################################

print('setting working directory')

# get current working directory
os.getcwd()

# set working directory
os.chdir('C:\\Users\\oisin_000\\Documents\\Kaggle\\Titanic Competition\\python')

#-- import data --#############################################################

print('loading data')

# load training set
train = pd.read_csv('./data/train.csv')

# load testing set
test = pd.read_csv('./data/test.csv')

#--combine the datasets --#####################################################

print('combining data')

# create a survived variable of nan values
test['Survived'] = np.full(418, np.nan)

# concate the datasets row wise
data = pd.concat(objs = [train, test], axis = 0)

# rest the column index
data = data.reset_index(drop = True)

#-- inital profiling --########################################################

# missing values
data.isnull().sum()

# column names
data.columns

# meta data
data.info()

# data types
data.dtypes

"""
###############################################################################
## Embarked ###################################################################
###############################################################################
"""

print('processing embarked')

# convert to a categorical variable
data.Embarked = data.Embarked.astype('category')

# check data type
data.Embarked.dtypes

# 2 nan values
data.Embarked.isnull().sum()

# impute the mode for the two missing values
data.Embarked = data.Embarked.fillna(data.Embarked.mode()[0])

# get levels of embarked variable
data.Embarked.cat.categories

"""
###############################################################################
## OrdinalEmbarked ############################################################
###############################################################################
"""

# create OrdinalEmbarked Variable
data['OrdinalEmbarked'] = data.Embarked.astype(str)

# create mapping levels
mp = {'S':1, 'C':2, 'Q':3}
data.OrdinalEmbarked = data.Embarked.map(mp).astype(int)

"""
###############################################################################
## Title ######################################################################
###############################################################################
"""

print('processing Title')

# no missing values
data.Name.isnull().sum()

# extract the title from the names
data['Title'] = data.Name.str.extract(pat = '(\w*,\s)([a-zA-Z\s]*.)', expand = False)[1]

# replace some of the title levels
data.Title = data.Title.replace(to_replace = ['Mme.'], value = 'Mrs.')
data.Title = data.Title.replace(to_replace = ['Mlle.', 'Ms.'], value  = 'Miss.')
data.Title = data.Title.replace(to_replace = ['Lady.', 'the Countess.','Capt.',
                                              'Col.','Don.', 'Dr.', 'Major.', 
                                              'Rev.', 'Sir.', 'Jonkheer.', 
                                              'Dona.'], 
                                value = 'Rare.')

# get the levels of title
data.Title.astype('category').cat.categories

# convert title and status to categories
data.Title = data.Title.astype('category')

"""
###############################################################################
## Ordinal Title ##############################################################
###############################################################################
"""

print('processing ordinal Title')

# create ordinaltitle variable
data['OrdinalTitle'] = data.Title.astype(str)

# create mapping
mp = {'Mr.':1, 'Miss.':2, 'Mrs.':3, 'Master.':4, 'Rare.':5}

# rank the status of the titles
data.OrdinalTitle = data.OrdinalTitle.map(mp).astype(int)

"""
###############################################################################
## Fare #######################################################################
###############################################################################
"""

print('processing fare')

# 1 nan values
data.Fare.isnull().sum()

# get the index of the missing value
data.Fare[data.Fare.isnull() == True].index

# get the values for the other attributes
data.iloc[1043, :]

# impute the mean for the missing values
mu = data[(data.Embarked == 'S') & (data.Sex == 'male') & (data.Pclass == 3) & (data.Title == 'Mr.')].Fare.median()

# impute the missing fare value
data.Fare = data.Fare.fillna(mu)

# plot fare as a histogram
data.Fare.plot(kind = 'hist', 
               title = 'Histogram of Fare', 
               grid = True)

# delete the excess list mu
del mu

"""
###############################################################################
## Ordinal Fare ###############################################################
###############################################################################
"""

print('processing ordinal fare')

# convert fare into a ordinal varible based on quantilies
data['OrdinalFare'] = pd.qcut(data['Fare'], q = 4)

# get the levels of OrdinalFare
data.OrdinalFare.cat.categories

# convert fare to a string
data['OrdinalFare'] = data['OrdinalFare'].astype(str)

# map the categories to an ordered ranks
mp = {'(-0.001, 7.896]':1, '(7.896, 14.454]':2, '(14.454, 31.275]':3, '(31.275, 512.329]':4}
data.OrdinalFare = data['OrdinalFare'].map(mp).astype(int)

"""
# use subsetting to map the levels accordingly
data.OrdinalFare[data.Fare <= 7.896] = 1
data.OrdinalFare[(data.Fare > 7.896) & (data.Fare <= 14.454)] = 2
data.OrdinalFare[(data.Fare > 14.454) & (data.Fare <= 31.275)] = 3
data.OrdinalFare[data.Fare > 31.275] = 4
"""

"""
###############################################################################
## Has Cabin ##################################################################
###############################################################################
"""

print('processing has cabin')

# create a has cabin indicator variable
data['HasCabin'] = data.Cabin.apply(lambda x: 0 if type(x) == float else 1)
data['HasCabin'] = data.Cabin.notnull().astype(int)

"""
###############################################################################
## Cabin ######################################################################
###############################################################################
"""

print('processing cabin')

# nan values
data.Cabin.isnull().sum()

# replace the nam values with missing
data.Cabin = data.Cabin.fillna('Missing')

# print the categories of cabin
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  print(data.Cabin.astype('category').cat.categories)

# replace the cabins with the floors
data.Cabin = data.Cabin.str.replace(pat = "(A\d*)", repl = "A")
data.Cabin = data.Cabin.str.replace(pat = "(B\d*)", repl = "B")
data.Cabin = data.Cabin.str.replace(pat = "(C\d*)", repl = "C")
data.Cabin = data.Cabin.str.replace(pat = "(D\d*)", repl = "D")
data.Cabin = data.Cabin.str.replace(pat = "(E\d*)", repl = "E")
data.Cabin = data.Cabin.str.replace(pat = "(F\d*)", repl = "F")
data.Cabin = data.Cabin.str.replace(pat = "(G\d*)", repl = "G")
data.Cabin = data.Cabin.str.replace(pat = "(^B[\sB]*)", repl = "B")
data.Cabin = data.Cabin.str.replace(pat = "(^C[\sC]*)", repl = "C")
data.Cabin = data.Cabin.str.replace(pat = "(^D[\sD]*)", repl = "D")
data.Cabin = data.Cabin.str.replace(pat = "(^E[\sE]*)", repl = "E")
data.Cabin = data.Cabin.str.replace(pat = "(F E)", repl = "F")
data.Cabin = data.Cabin.str.replace(pat = "(F G*)", repl = "F")

# create a floor variable
data['Floor'] = data.Cabin

# conbert floor to a category
data.Floor = data.Floor.astype('category')

# drop Cabin varialbe
data = data.drop(labels = 'Cabin', axis = 1)

"""
###############################################################################
## Ticket #####################################################################
###############################################################################
"""

print('processing ticket')

#-- Ticket Prefix --###########################################################

# no nan values
data.Ticket.isnull().sum()

# create prefix variable
data['TicketPrefix'] = data.Ticket

# remove the suffix number
data['TicketPrefix'] = data.TicketPrefix.str.replace(pat = '(\d*)$', repl = '')
data['TicketPrefix'] = data.TicketPrefix.str.replace(pat = '(\s*)', repl = '')

# replace any empty values with nan
data.TicketPrefix[data.TicketPrefix == ''] = 'Missing'

# print the categories of prefix
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  print(data.TicketPrefix.astype('category').cat.categories)

# convert prefix to a category
data.TicketPrefix = data.TicketPrefix.astype('category')

#-- Ticket Suffix --###########################################################

# create suffix variab;e
data['TicketSuffix'] = data.Ticket

# extract the ticket number
data.TicketSuffix = data.TicketSuffix.str.extract(pat = '(\d*)$', expand = False)

# replace any empty values with 0
data.TicketSuffix[data.TicketSuffix == ''] = 0

# convert the Suffix to a float
data.TicketSuffix = data.TicketSuffix.astype(int)

"""
###############################################################################
## Sex ########################################################################
###############################################################################
"""

print('processing sex')

# no nan values
data.Sex.isnull().sum()

# convert the Suffix to a float
data.Sex = data.Sex.astype('category')

"""
###############################################################################
## Male #######################################################################
###############################################################################
"""

print('processing Male')

# create a binary indicator for male
data['Male'] = data.Sex.apply(lambda x: 1 if x == 'male' else 0)
data['Male'] = (data.Sex == 'male').astype(int)
# data['Male'] = data.Sex.map({'male':1, 'female':0}).astype(int)

"""
###############################################################################
## Family Size ################################################################
###############################################################################
"""

print('processing family size')

# create a family size variable
data['FamilySize'] = data.Parch + data.SibSp

"""
###############################################################################
## Is Alone ###################################################################
###############################################################################
"""

print('processing is alone')

# create using boolean arguments
data['IsAlone'] = (data.FamilySize == 0).astype(int)

# create by applying a lambda function
data['IsAlone'] = data.FamilySize.apply(lambda x: 1 if x == 0 else 0)

"""
###############################################################################
## Age ########################################################################
###############################################################################
"""

print('processing age')

"""
there are a couple of options for processing the missing values in age
1 - impute random integers between plus or minus 1 standard deviation
"""

# output the dataset for processing_age script
data.to_csv('./data/data_pII.csv')

# build a seperate model to predict age using similar methods
# either call the script at this point 
# or read out the current results and import the derived results

#-- impute random integers --##################################################

# plot a histogram of age
data.Age.plot(kind = 'hist',
              title = 'Histogram of Age',
              grid = True)

# plot a boxplot of age
data.Age.plot(kind = 'box',
              title = 'Histogram of Age',
              grid = True)

# impute a random integer for the ages
imAge = np.random.randint(low = data.Age.mean() - data.Age.std(),
                          high = data.Age.mean() + data.Age.std(),
                          size = data.Age.isnull().sum())
data.Age[data.Age.isnull()] = imAge

# delete excess data
del imAge

"""
###############################################################################
## OrdinalAge #################################################################
###############################################################################
"""

print('processing ordinal Age')

# note can categorise data based on equal bin width and equal quantilies

# convert Age into an ordinal variable based on quantiles
data['OrdinalAge'] = pd.qcut(data.Age, q = 5)

# get the levels of OrdinalFare
data.OrdinalAge.cat.categories

# convert ordinal age to a string
data['OrdinalAge'] = data['OrdinalAge'].astype(str)

# map the categories to an ordered ranks
data.OrdinalAge = data.OrdinalAge.map({'(0.169, 19.0]':1,
                                        '(19.0, 25.0]':2,
                                        '(25.0, 31.0]':3,
                                        '(31.0, 40.0]':4,
                                        '(40.0, 80.0]':5})

"""
#########################
## Output Data Stage 1 ##
#########################
"""

# save the cleaned training data
clean_train = data[data.Survived.notnull()]
clean_train.to_csv('.\data\clean_train.csv')
  
# save the cleaned testing data
clean_test = data[data.Survived.isnull()]
clean_test.to_csv('.\data\clean_test.csv')

"""
###############################################################################
## Feature Reduction ##########################################################
###############################################################################
"""

# get the column names
data.columns

# drop irrelavent data files
data = data.drop(labels = ['PassengerId', 'Name', 'Ticket', 'Sex'], axis = 1)

"""
#-- drop categorical variables --##############################################

# get the data types
data.dtypes

# drop passenger id from the dataset
data = data.drop(labels = ['PassengerId', 'Name', 'Ticket', 'Embarked', 'Sex',
                           'Title', 'Floor', 'TicketPrefix'], axis = 1)
"""

"""
#########################
## Output Data Stage 2 ##
#########################
"""

# save the cleaned training data
clean_train = data[data.Survived.notnull()]
clean_train.to_csv('.\data\clean_train.csv')
  
# save the cleaned testing data
clean_test = data[data.Survived.isnull()]
clean_test.to_csv('.\data\clean_test.csv')


"""
###############################################################################
## Dummy Encode Categorical Variables #########################################
###############################################################################
"""

print('dummy encode categorical variables')

# dummy encode the categorical variables
data = pd.get_dummies(data)

"""
###############################################################################
## Derive Interaction and Polynomial Terms ####################################
###############################################################################
"""

print('generate interaction and polynomial terms')

# extract predictors
derive_data = data.drop(labels = 'Survived', axis = 1)

# initiate interaction and polynomials
poly = preprocessing.PolynomialFeatures()

# fit poly terms
poly_data = poly.fit_transform(derive_data)

# save column names
col_names = poly.get_feature_names(derive_data.columns)

# turn poly df 
derive_data = pd.DataFrame(poly_data, columns = col_names)

# check for null values
derive_data.isnull().any().any()

# reconstruct the dataset
data = pd.concat(objs = [data.Survived, derive_data], axis = 1)

# delete the excess data
del derive_data, col_names, poly_data

"""
#########################
## Output Data Stage 3 ##
#########################
"""

# save the cleaned training data
clean_train = data[data.Survived.notnull()]
clean_train.to_csv('.\data\clean_train.csv')
  
# save the cleaned testing data
clean_test = data[data.Survived.isnull()]
clean_test.to_csv('.\data\clean_test.csv')

"""
###############################################################################
## Feature Selection ##########################################################
###############################################################################
"""

#-- select best 100 terms --###################################################

# Not these univariate selection techniques only work on complete datasets

# extract the selection data
select_data = data[data.Survived.notnull()]
select_Survived = data.Survived[data.Survived.notnull()].astype('category')

# intiate select top 100 attributes
selector = feature_selection.SelectKBest(score_func = feature_selection.chi2, 
                                         k = 50)

# select the attributes
s_data = selector.fit_transform(select_data, select_Survived)

# get colnames for the selection data
col_names = select_data.columns[selector.get_support(indices = True)]

# save the selected features as a dataset
select_data = pd.DataFrame(select_data, columns = col_names)

# save the column names 
col_names = select_data.columns

# extract the best columns from the incomplete data
select_data = data.loc[:, col_names]

# reconstruct the dataset
data = pd.concat(objs = [data.Survived, select_data], axis = 1)

# delete the excess data
del select_Survived, select_data, s_data

"""
#########################
## Output Data Stage 4 ##
#########################
"""

# save the cleaned training data
clean_train = data[data.Survived.notnull()]
clean_train.to_csv('.\data\clean_train.csv')
  
# save the cleaned testing data
clean_test = data[data.Survived.isnull()]
clean_test.to_csv('.\data\clean_test.csv')

"""
###############################################################################
## Scale the data #############################################################
###############################################################################
"""

# extract the data to be standardised
stand_data = data.drop(labels = ['Survived'], axis = 1)

# initiate the range standardisation
scalar = preprocessing.MinMaxScaler().fit(stand_data)

# perform the standardisation
scalar_data = scalar.transform(stand_data)

# save the scaled data as a dataframe
scalar_data = pd.DataFrame(scalar_data, columns = stand_data.columns)

# attact the target variable survived
scalar_data = pd.concat(objs = [data.Survived, scalar_data], axis = 1)

# overwrite the data dataframe
data = scalar_data

# delete excess data
del scalar_data, stand_data

"""
#########################
## Output Data Stage 5 ##
#########################
"""

# save the cleaned training data
clean_train = data[data.Survived.notnull()]
clean_train.to_csv('.\data\clean_train.csv')
  
# save the cleaned testing data
clean_test = data[data.Survived.isnull()]
clean_test.to_csv('.\data\clean_test.csv')

"""
###############################################################################
## Oversample Target Variable #################################################
###############################################################################
"""

# need to balance the distribution of the target variable
s = (data.Survived == 0).sum() - (data.Survived == 1).sum()

# generate random samples from the index of survivors
idx = np.random.choice(data[data.Survived == 1].index, 
                       size = s,
                       replace = False)

# sample the data
sampled_data = data.iloc[idx, :]

# reconstruct the data
data = pd.concat(objs = [data, sampled_data], axis = 0)

# randomise the final dataset
r = data.shape[0]
idx = np.random.choice(a = range(r),
                       size = r,
                       replace = False)
data = data.iloc[idx, :]

# delete excess data
del idx, r, s

"""
#########################
## Output Data Stage 6 ##
#########################
"""

# save the cleaned training data
clean_train = data[data.Survived.notnull()]
clean_train.to_csv('.\data\clean_train.csv')
  
# save the cleaned testing data
clean_test = data[data.Survived.isnull()]
clean_test.to_csv('.\data\clean_test.csv')