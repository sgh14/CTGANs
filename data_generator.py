import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt


def generate_data(latent_dim, labels, generator_path='models/generator.h5'):
  generator = models.load_model(generator_path)
  batch_size = len(labels['energy'])
  random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
  random_vector_labels = tf.concat(
      [random_latent_vectors,
      np.reshape(labels['particletype'], (-1, 2)),
      np.reshape(labels['energy'], (-1, 1)),
      np.reshape(labels['direction'], (-1, 2))],
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
    # img = keras.preprocessing.image.array_to_img(img)
    # img.save(f'digit_{i}.png')