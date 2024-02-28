"""
Contains TensorFlow code for Cat vs Dogs CNN model
"""
import tensorflow as tf

def build_model(random_seed=42):
  tf.random.set_seed(random_seed)

  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3,3),
                           activation='relu',
                           input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3,3),
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3,3),
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),

    # output layer
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# using Adam
#model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
#                optimizer=tf.keras.optimizers.Adam(learng_rate=1e-3),
#                metrix=['accuracy'])

# using RMSprop
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                metrics=['accuracy'])

  return model
