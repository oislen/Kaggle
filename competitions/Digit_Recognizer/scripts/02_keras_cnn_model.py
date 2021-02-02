# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:21:47 2021

@author: oislen
"""

# load in relevant libraries
import cons
import numpy as np
import pandas as pd
from CNN.LeNet_Model import LeNet_Model
from fit_model import fit_model
from process_data import process_data

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop #, Adam
from graph import plot_images, plot_confusion_matrix, plot_errors
from sklearn.metrics import confusion_matrix

# run process data func to load and process data
X_train, y_train, X_valid, y_valid, X_test = process_data(train_data_fpath = cons.train_data_fpath,
                                                          test_data_fpath = cons.test_data_fpath,
                                                          valid_size = cons.valid_size,
                                                          random_state = cons.random_state
                                                          )

# plot training and validation data
plot_images.plot_images(X_train)
plot_images.plot_images(X_valid)
plot_images.plot_images(X_test)

# define image augmentation
datagen = ImageDataGenerator(horizontal_flip = True,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             zoom_range = 0.1,
                            )

# apply image augmentation
datagen.fit(X_train)

################
#-- Modellig --#
################

# generate lenet model architecture
model = LeNet_Model(input_shape = cons.sample_shape,
                    n_targets = 10
                    )

# define the optimiser to use
#optimizer = Adam(lr = 0.001)
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
    
# Attention: Windows implementation may cause an error here. In that case use model_name=None.
fit_model(model_name = 'lenet', 
          model = model, 
          epochs = 1,
          batch_size = cons.batch_size,
          optimizer = optimizer,
          datagen = datagen, 
          X_train = X_train,
          X_val = X_valid, 
          Y_train = y_train, 
          Y_val = y_valid
          )

#########################
#-- Model Predicitons --#
#########################

# Predict the values from the validation dataset
Y_pred = model.predict(X_valid)

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors
Y_true = np.argmax(y_valid,axis = 1) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
plot_confusion_matrix.plot_confusion_matrix(confusion_mtx)

######################
#-- Error Analysis --#
###################### 

# find error cases
errors = (Y_pred_classes - Y_true != 0)

# extract error cases
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_valid[errors]

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
plot_errors.plot_errors(most_important_errors, 
                        X_val_errors, 
                        Y_pred_classes_errors, 
                        Y_true_errors
                        )

#########################
#-- Kaggle Submission --#
#########################

# predict results
results = model.predict(X_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv(cons.pred_data_fpath,
                  index = False
                  )

