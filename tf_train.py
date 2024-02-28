"""
Training code for TF model
"""
import tensorflow as tf

def tf_model_training(model, ds_train, ds_validation, epochs)
  history = model.fit(ds_train_ds,
                    epochs=epochs,
                    steps_per_epoch=len(ds_train),
                    validation_data=ds_validataion,
                    validation_steps=len(ds_validataion),
                    verbose=2)
  return history
  
