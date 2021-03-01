# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:24:04 2019

@author: oislen
"""

    
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import numpy as np

class StackModels(BaseEstimator, RegressorMixin, TransformerMixin):
    
    """
    
    Stacked Models Class
    
    """
    
    ################
    #-- __init__ --#
    ################
    
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    ###########
    #-- Fit --#
    ###########
    
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        
        """
        Fit Models Documentation
        
        Function Overview
        
        This function fits the various sklearn models to the given datasets.
        
        Parameters
        
        self - Sklearn Class, the class object
        X - Numpy Array, the predictor variables
        y - Numpy Array, the response variable
        
        Example
        
        StackMod.fit(X_train_full, y_train_full)
        
        """
        
        # create empty lists for each model
        self.base_models_ = [list() for x in self.base_models]
        
        # clone the meta model
        self.meta_model_ = clone(self.meta_model)
        
        # create k-folds to
        kfold = KFold(n_splits = self.n_folds, 
                      shuffle = True, 
                      random_state = 1234
                      )
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # for each model
        for i, model in enumerate(self.base_models):
            
            # for the the training and validation folds indexes
            for train_index, holdout_index in kfold.split(X, y):
                
                # clone an instance of the model
                instance = clone(model)
                
                # append this instance to the base models
                self.base_models_[i].append(instance)
                
                # fit the instance of the model to the training indexs
                instance.fit(X[train_index], y[train_index])
                
                # predict for the validation indexes
                y_pred = instance.predict(X[holdout_index])
                
                # map these validation predictions across to the out of fold predictions
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    ###############
    #-- Predict --#
    ###############
    
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        
        """
        Fit Models Documentation
        
        Function Overview
        
        This function fits the various sklearn models to the given datasets.
        
        Parameters
        
        self - Sklearn Class, the class object
        X - Numpy Array, the variables to predict for
        
        Example
        
        StackMod.predict(X_test)
        
        """
        
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
