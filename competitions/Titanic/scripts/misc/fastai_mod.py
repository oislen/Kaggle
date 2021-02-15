# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:34:48 2021

@author: oislen
"""

import os
import sys
scripts_dir = os.path.dirname(os.getcwd())
sys.path.append(scripts_dir)

import cons

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# linear algebra
import numpy as np 
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
import fastai
fastai.__version__ 

from fastai import *
from fastai.tabular.all import *

# set pandas behaviour
pd.options.display.max_rows = 20
pd.options.display.max_columns = None

# load in train and test sets
df = pd.read_csv(cons.train_data_fpath, low_memory=False)
df_test = pd.read_csv(cons.test_data_fpath, low_memory=False)

# RF does not need `Normalize`
procs = [Categorify, FillMissing] 

splits = RandomSplitter(valid_pct=0.2)(range_of(df))

dep_var='Survived'

cont,cat = cont_cat_split(df, 1, dep_var=dep_var)

cont

cat

to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits, y_block=CategoryBlock())

to.show(3)

X_train, y_train = to.train.xs,to.train.y
X_valid, y_valid = to.valid.xs,to.valid.y

from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=100, n_jobs=-1)

m.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred=m.predict(X_valid)

accuracy_score(y_valid, y_pred)

to_test = TabularPandas(df_test, procs, cat, cont)

# remove reduntant columns (training did not use this col)
predicted_result = m.predict(to_test.xs.drop('Fare_na', axis=1)) 

output= pd.DataFrame({'PassengerId':df_test.PassengerId, 'Survived': predicted_result.astype(int)})
output.to_csv('submission_titanic.csv', index=False)
output.head()