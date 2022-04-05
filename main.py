#%%
import yaml
from tensorflow.keras import models

from data_loader import load_data
from predictor import get_predictor
from discriminator import Discriminator, get_discriminator_loss, get_discriminator_optimizer
from generator import Generator, get_generator_loss, get_generator_optimizer
from GANs import GANs
from callback import Checkpoint


config_path = input('Configuration file path: ') #'config_files/GANs.yml'
with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

g_config = config['Generator']
d_config = config['Discriminator']
gans_config = config['GANs']

#%% LOAD DATA
dataset = load_data(config, **config['Input'])

#%% TRAIN THE CTLearn MODEL IF THERE ISN'T ANY ALREADY SAVED
predictor = get_predictor(**config['Predictor'])

#%% BUILD THE GENERATOR
g_path = g_config['predefined_model_path']
generator = models.load_model(g_path) if g_path else Generator(g_config)

#%% BUILD THE DISCRIMINATOR
d_path = d_config['predefined_model_path']
discriminator = models.load_model(d_path) if d_path else Discriminator(d_config)

#%% BUILD GANS
# Instantiate the GANs model.
gans = GANs(
    dataset=dataset,
    discriminator=discriminator,
    generator=generator,
    predictor=predictor,
    discriminator_steps=gans_config['discriminator_steps'],
    generator_steps=gans_config['generator_steps'],
    gp_weight=gans_config['gp_weight']
)

# Instantiate the optimizer for both networks
generator_optimizer = get_generator_optimizer(**g_config['optimizer'])
discriminator_optimizer = get_discriminator_optimizer(**d_config['optimizer'])

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
checkpoint = Checkpoint(dataset, **config['Callback'])
history = gans.fit(dataset, epochs=gans_config['epochs'], verbose=1, callbacks=[checkpoint])

