import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# def plot_grid(images, nrows=5, save_path=''):
#   channels = images.shape[-1]
#   fig, axes = plt.subplots(nrows, channels, figsize=(channels*2, nrows*2), squeeze=False)
#   for i in range(axes.shape[0]):
#     for j in range(axes.shape[1]):
#       axes[i, j].pcolor(np.squeeze(images[i, :, :, j]), cmap='viridis')
#       axes[i, j].axis('off')
      
#   for j in range(axes.shape[1]):
#     axes[0, j].set_title(f'Channel {j+1}')
  
#   if save_path:
#     fig.savefig(save_path)

def plot_grid(images, nrows=5, ncols=5, save_path=''):
  channels = images.shape[-1]
  fig = plt.figure(constrained_layout=True, figsize=(ncols*channels*2, nrows*2))
  subfigs = fig.subfigures(nrows=nrows, ncols=ncols, squeeze=False, wspace=0.1, hspace=0.1)
  for i in range(nrows):
    for j in range(ncols):
      axes = subfigs[i, j].subplots(nrows=1, ncols=channels, squeeze=False)
      axes = axes.flatten()
      for k in range(channels):
        axes[k].pcolor(np.squeeze(images[i*ncols+j, :, :, k]), cmap='viridis')
        axes[k].axis('off')
        if i == 0:
          axes[k].set_title(f'Channel {k+1}')
  
  if save_path:
    fig.savefig(save_path)