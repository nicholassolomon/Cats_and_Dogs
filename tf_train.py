"""
Training code for TF model
"""
import tensorflow as tf

def tf_model_training(model, 
  ds_train, ds_validation, epochs, verbose=2):
  history = model.fit(ds_train,
                    epochs=epochs,
                    steps_per_epoch=len(ds_train),
                    validation_data=ds_validation,
                    validation_steps=len(ds_validation),
                    verbose=verbose)
  return history
  
