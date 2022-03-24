#%%
from tensorflow.keras import optimizers, models
import os

from data_loader import *
from predictor import *
from discriminator import Discriminator, get_discriminator_loss
from generator import Generator, get_generator_loss
from GANs import *
from data_generator import *


config_path = 'config_files/GANs.yml'
with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

#%% LOAD DATA
dataset = load_data(config, **config['Input'])

#%% TRAIN THE CTLearn MODEL IF THERE ISN'T ANY ALREADY SAVED
predictor = get_predictor(**config['Predictor'])

#%% BUILD THE GENERATOR
g_path = config['Generator']['predefined_model_path']
generator = models.load_model(g_path) if g_path else Generator(config['Generator'])

#%% BUILD THE DISCRIMINATOR
d_path = config['Discriminator']['predefined_model_path']
discriminator = models.load_model(d_path) if d_path else Discriminator(config['Discriminator'])

#%% BUILD GANS
# Instantiate the GANs model.
gans = GANs(
    discriminator=discriminator,
    generator=generator,
    predictor=predictor,
    discriminator_extra_steps=config['GANs']['discriminator_extra_steps'], # TODO: should be 0 for no extra steps
    generator_extra_steps=config['GANs']['generator_extra_steps'],
    gp_weight=config['GANs']['gp_weight']
)

# Instantiate the optimizer for both networks
generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9) 

# Get loss functions
generator_loss = get_generator_loss(**config['Generator']['loss'])
discriminator_loss = get_discriminator_loss(**config['Discriminator']['loss'])

# Compile the GANs model.
gans.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)

#%% TRAIN GANS
plot_and_save = Plot_and_save(dataset, **config['Callback'])
history = gans.fit(dataset, epochs=config['GANs']['epochs'], verbose=1, callbacks=[plot_and_save])
