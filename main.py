#%%
import tensorflow as tf
from tensorflow.keras import models, optimizers
import os

from data_loader import *
from predictor import *
from discriminator import get_discriminator, get_discriminator_loss
from generator import get_generator, get_generator_loss
from GANs import *
from data_generator import *

#%% LOAD DATA
config_files_dir = 'config_files'
GANs_config = os.path.join(config_files_dir, 'GANs.yml')
dataset, data_features = load_data(GANs_config, 'train', batch_size=64, shuffle=False)
latent_dim = 512

#%% TRAIN THE CTLearn MODEL IF THERE ISN'T ANY ALREADY SAVED
predictor_config = os.path.join(config_files_dir, 'predictor.yml')
predictor = get_predictor(predictor_config)

#%% BUILD THE GENERATOR
generator_in_shape = (latent_dim + data_features['labels_dim'],)
generator = get_generator(generator_in_shape)

#%% BUILD THE DISCRIMINATOR
discriminator_in_shape = data_features['image_shape']
discriminator = get_discriminator(discriminator_in_shape)

#%% BUILD GANS
# Instantiate the GANs model.
gans = GANs(
    discriminator=discriminator,
    generator=generator,
    predictor=predictor,
    latent_dim=latent_dim,
    discriminator_extra_steps=1,
    generator_extra_steps=1
)

# Instantiate the optimizer for both networks
generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9) 

# Get loss functions
generator_loss = get_generator_loss()
discriminator_loss = get_discriminator_loss()

# Compile the GANs model.
gans.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)

#%% TRAIN GANS
# tensorboard = callbacks.TensorBoard('./logs', update_freq=1)
plot_and_save = Plot_and_save(dataset)
history = gans.fit(dataset, epochs=2, verbose=1, callbacks=[plot_and_save])
