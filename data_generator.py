import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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