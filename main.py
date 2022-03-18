#%%
from tensorflow.keras import optimizers
import os

from data_loader import *
from predictor import *
from discriminator import Discriminator, get_discriminator_loss
from generator import Generator, get_generator_loss
from GANs import *
from data_generator import *

#%% LOAD DATA
config_files_dir = 'config_files'
GANs_config = os.path.join(config_files_dir, 'GANs.yml')
dataset = load_data(GANs_config, 'train', batch_size=64, shuffle=False)

#%% TRAIN THE CTLearn MODEL IF THERE ISN'T ANY ALREADY SAVED
predictor_config = os.path.join(config_files_dir, 'predictor.yml')
predictor = get_predictor(predictor_config)

#%% BUILD THE GENERATOR
g_path = None
generator = models.load_model(g_path) if g_path else Generator(latent_dim=512)

#%% BUILD THE DISCRIMINATOR
d_path = None
discriminator = models.load_model(d_path) if d_path else Discriminator()

#%% BUILD GANS
# Instantiate the GANs model.
gans = GANs(
    discriminator=discriminator,
    generator=generator,
    predictor=predictor,
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
