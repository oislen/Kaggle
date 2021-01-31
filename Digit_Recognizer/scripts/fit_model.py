# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:31:42 2021

@author: oislen
"""

import os
from keras.optimizers import RMSprop #, Adam
from keras.callbacks import ModelCheckpoint
from graph import plot_history

def fit_model(model,
              datagen, 
              X_train, 
              Y_train,
              X_val, 
              Y_val, 
              batch_size,
              model_name=None,
              output_dir="data/checkpoints",
              class_weight=None, 
              epochs=60, 
              lr=0.001, 
              verbose=False
              ):
    """
    
    Fit model.
    
    You can edit this function anyhow.
    
    """
    
    if verbose:
        model.summary()
    
    # define the optimiser to use
    #optimizer = Adam(lr = lr)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    # compile model (can use another optimizer)
    model.compile(optimizer = optimizer,#Adam(lr=lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy']
                 )
    
    # starts training
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        validation_data = (datagen.standardize(X_val), Y_val),
                        epochs = epochs, 
                        steps_per_epoch=len(X_train) // batch_size,
                        callbacks = [ModelCheckpoint(os.path.join(output_dir, "{model_name}").format(model_name=model_name) + "-{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True),
                                     ] if model_name is not None else [],
                        class_weight = class_weight
                        )  
    
    plot_history(history)