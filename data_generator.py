import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt


def generate_data(latent_dim, labels, generator_path='models/generator.h5'):
  generator = models.load_model(generator_path)
  batch_size = len(labels['energy'])
  random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
  random_vector_labels = random_latent_vectors
  for task in labels.values():
    label_shape = task.shape[1] if np.ndim(task) == 2 else 1
    random_vector_labels = tf.concat(
      [random_vector_labels, np.reshape(task, (-1, label_shape))],
      axis = 1
    )

  generated_images = generator(random_vector_labels).numpy()
  
  return generated_images


def plot_image(image, save_path=''):
  fig, ax = plt.subplots(figsize=(8,8))
  ax.pcolor(np.squeeze(image), cmap='viridis')
  ax.axis('off')
  if save_path:
    fig.savefig(save_path)


def plot_grid(images, nrows=5, ncols=5, save_path=''):
  fig, axes = plt.subplots(nrows, ncols, figsize=(10,10))
  axes = axes.ravel()
  for k, ax in enumerate(axes):
    ax.pcolor(np.squeeze(images[k]), cmap='viridis')
    ax.axis('off')
  
  if save_path:
    fig.savefig(save_path)