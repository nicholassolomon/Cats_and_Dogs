"""
Utilities when evaluating model and reviewing picture data
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pathlib
import numpy as np
import os
import tensorflow as tf


def tf_plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


def get_classes(dir_path):
  """
  Get a list of class names based on the folder names in the image subdirectories
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    List of directory names as class names
  """
  # Get the class names (programmatically, this is much more helpful with a longer list of classes)
  data_dir = pathlib.Path(data_path) # turn our training path into a Python path
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
  return class_names

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def view_random_image(target_dir, target_class):
  """
  Select a random image from target_dir and display
  """
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

def get_min_max_shapes(image_dir):
  """
  This function takes a target directory containing images and returns the minimum and maximum shapes of the images.

  Args:
    target_dir: The directory containing the images.

  Returns:
    min_shape: A tuple containing the minimum height, width, and number of channels of the images.
    max_shape: A tuple containing the maximum height, width, and number of channels of the images.
  """
  min_shape = (np.inf, np.inf, np.inf)
  max_shape = (0, 0, 0)
  
  for filename in os.listdir(image_dir):
    img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, filename))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    min_shape = tuple(min(min_shape[i], img_arr.shape[i]) for i in range(3))
    max_shape = tuple(max(max_shape[i], img_arr.shape[i]) for i in range(3))

  return min_shape, max_shape
  
