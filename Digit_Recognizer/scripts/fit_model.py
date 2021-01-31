# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:31:42 2021

@author: oislen
"""

import os
from keras.callbacks import ModelCheckpoint
from graph import plot_history

def fit_model(model,
              X_train, 
              Y_train,
              X_val, 
              Y_val, 
              batch_size,
              optimizer,
              datagen = None, 
              model_name = None,
              output_dir = "data/checkpoints",
              class_weight = None, 
              epochs = 60, 
              verbose = False
              ):
    """
    
    Fit model.
    
    You can edit this function anyhow.
    
    """
    
    if verbose:
        model.summary()
    
    # compile model (can use another optimizer)
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy']
                 )
    
    if datagen == None:
        validation_data = (X_val, Y_val)
        x = X_train 
        y = Y_train
    else:
        validation_data = (datagen.standardize(X_val), Y_val)
        x = datagen.flow(X_train, Y_train, batch_size=batch_size)
        y = None
    
    # calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    
    # generate checkpoints
    check_points = [ModelCheckpoint(os.path.join(output_dir, "{model_name}").format(model_name=model_name) + "-{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True),] if model_name is not None else []
    
    # starts training
    history = model.fit(x = x,
                        validation_data = validation_data,
                        y = y,
                        epochs = epochs, 
                        steps_per_epoch = steps_per_epoch,
                        callbacks = check_points,
                        class_weight = class_weight
                        )  
    
    plot_history(history)
    
    return 0