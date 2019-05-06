# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:26:42 2018
@author: oisin_000
"""

"""
###############################################################################
## Preliminaries ##############################################################
###############################################################################
Table of Contents
Embarked
Title
Fare
Cabin
Ticket
Sex
Male
Family Size
Alone
Age
Feature Reduction
Dummy Encode Categorical Variables
Derive Polynomial and Interaction Terms
Feature Selection
Principle Components Analysis
Standardise Data
Split the Data
Oversample Target
Output Data
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
from sklearn import cluster
from sklearn import tree

# statsmodels will be used for principle components analysis
from statsmodels.multivariate import pca

#-- set working directory --###################################################

print('setting working directory')

# get current working directory
os.getcwd()

# set working directory
#os.chdir('C:\\Users\\oisin_000\\Documents\\Kaggle\\Titanic Competition\\python')
os.chdir('C:\\Users\\LeonOi01\\Downloads')

#-- import data --#############################################################

print('loading data')

# load training set
train = pd.read_csv('./data/train.csv')
train = pd.read_csv('train.csv')

# load testing set
test = pd.read_csv('./data/test.csv')
test = pd.read_csv('test.csv')

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
## Cabin ######################################################################
###############################################################################
"""

print('processing has cabin')

# create a has cabin indicator variable
data['HasCabin'] = data.Cabin.apply(lambda x: 0 if type(x) == float else 1)
data['HasCabin'] = data.Cabin.notnull().astype(int)

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
## Alone ######################################################################
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
# data.to_csv('./data/data_pII.csv')

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
###############################################################################
## Feature Reduction ##########################################################
###############################################################################
Feature reductions allows for the removal of irrelavent variables.
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
###############################################################################
## Dummy Encode Categorical Variables #########################################
###############################################################################
Dummy encoding categorical variables allows the for the modelling of categorical 
variables.
"""

print('dummy encode categorical variables')

# dummy encode the categorical variables
data = pd.get_dummies(data)

"""
###############################################################################
## Derive Interaction and Polynomial Terms ####################################
###############################################################################
The derivation of higher dimenionsal and interaction terms allows for the 
modelling of non-linear trends in the data.
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
###############################################################################
## Feature Selection ##########################################################
###############################################################################
Feature selection determines the top K predictive variables in the dataset.
"""

#-- select the top 10% attributes --###########################################

# Not these univariate selection techniques only work on complete datasets

# extract the selection data
select_data = data[data.Survived.notnull()]
select_Survived = data.Survived[data.Survived.notnull()].astype('category')

# drop Survived from the select data
select_data = select_data.drop(labels = 'Survived', axis = 1)

# intiate select top 100 attributes
selector = feature_selection.SelectKBest(score_func = feature_selection.chi2, 
                                         k = int(np.round(select_data.shape[1] * 0.1)))

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
###############################################################################
## Priciple Components Analysis ###############################################
###############################################################################
Principle components analysis reduces the dimension by removing dimensions which
explain little variance in the data.
"""

# drop the survived variable as it has nan values
pca_data = data.drop(labels = 'Survived', axis = 1)

# perform the pca 
principle = pca.PCA(data = pca_data, standardize = True)

# return the rsq from each of the principle components
principle.rsquare

# the principle components
pca_data = pd.DataFrame(principle.scores)

# select the dimension
pca_data = pca_data.iloc[:, 0:int(len(principle.rsquare[principle.rsquare <= 0.9]))]

# reconstruct the dataset
data = pd.concat(objs = [data.Survived, pca_data], axis = 1)

# delete the excess data
del pca_data

"""
###############################################################################
## Standardise Data ###########################################################
###############################################################################
Standardising data allows for scales the dimensions to a common magnitude.
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
###############################################################################
## Clustering #################################################################
###############################################################################
"""

# extract the cluster data
clust_data = data.drop(labels = ['Survived'], axis = 1)

# initiate the agglomerative clustering
agglom = cluster.AgglomerativeClustering(n_clusters = 2, 
                                         affinity = 'euclidean')

# fit the agglomerative clustering to the dataset
mod = agglom.fit(clust_data)

# extract the labels from the cluster data
data['cluster'] = mod.labels_

# delete the excess data
del clust_data

"""
###############################################################################
## Feature Selection ##########################################################
###############################################################################
"""

#-- select attributes based on importance within decision tree --##############

# Not these univariate selection techniques only work on complete datasets

# extract the selection data
select_data = data[data.Survived.notnull()]
select_Survived = data.Survived[data.Survived.notnull()].astype('category')

# extract the selection data
select_data = select_data.drop(labels = 'Survived', axis = 1)

# intiate select top 100 attributes
selector = feature_selection.SelectFromModel(estimator = tree.DecisionTreeClassifier(), 
                                             threshold = "mean")

# select the attributes
s_data = selector.fit_transform(X = select_data, y = select_Survived)

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
###############################################################################
## Split the Dataset ##########################################################
###############################################################################
Splitting the dataset stores a hold out test set 
"""

# split the data between training and test sets
train_data = data[data.Survived.notnull()]
test_data = data[data.Survived.isnull()]

# split the data 70:30 training:validation
train_data = train_data.iloc[0:int(np.round(train_data.shape[0]*0.7)), :]
valid_data = train_data.iloc[int(np.round(train_data.shape[0]*0.7)):, :]

"""
###############################################################################
## Oversample Target Variable #################################################
###############################################################################
"""

# need to balance the distribution of the target variable
s = (train_data.Survived == 0).sum() - (train_data.Survived == 1).sum()

# generate random samples from the index of survivors
idx = np.random.choice(train_data[train_data.Survived == 1].index, 
                       size = s,
                       replace = False)

# sample the data
sampled_data = train_data.iloc[idx, :]

# reconstruct the data
train_data = pd.concat(objs = [train_data, sampled_data], axis = 0)

# randomise the final dataset
r = train_data.shape[0]
idx = np.random.choice(a = range(r),
                       size = r,
                       replace = False)
train_data = train_data.iloc[idx, :]

# delete excess data
del idx, r, s, sampled_data

"""
###############################################################################
## Output Data Sets ###########################################################
###############################################################################
"""

# save the cleaned training data
train_data.to_csv('.\data\clean_train.csv',
                  index = False)

# save the cleaned validation data
valid_data.to_csv('.\data\clean_valid.csv',
                  index = False)

# save the cleaned testing data
test_data.to_csv('.\data\clean_test.csv',
                  index = False)