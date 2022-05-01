import yaml
from tensorflow.keras import models

from data_loader import load_data
from predictor import get_predictor
from discriminator import Discriminator, get_discriminator_loss, get_discriminator_optimizer
from generator import Generator, get_generator_loss, get_generator_optimizer
from GANs import GANs
from callback import Checkpoint


# Load the config file
config_path = input('Configuration file path: ') #'config_files/GANs.yml'
with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

# Get the configurations for the generator, the discriminator and the GANs model
g_config = config['Generator']
d_config = config['Discriminator']
gans_config = config['GANs']

# Load the data
dataset = load_data(config, **config['Input'])

# Train the predcitor (CTLearn auxiliary model) there isn't any already trained
predictor = get_predictor(**config['Predictor'])

# Build the generator or load a predefined one
g_path = g_config['predefined_model_path']
generator = models.load_model(g_path) if g_path else Generator(g_config)

# Build the discriminator or load a predefined one
d_path = d_config['predefined_model_path']
discriminator = models.load_model(d_path) if d_path else Discriminator(d_config)

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

# Get loss functions for both networks
generator_loss = get_generator_loss(**g_config['loss'])
discriminator_loss = get_discriminator_loss(**d_config['loss'])

# Compile the GANs model.
gans.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)

# Instantiate the callback and train the GANs
checkpoint = Checkpoint(dataset, **config['Callback'])
history = gans.fit(dataset, epochs=gans_config['epochs'], verbose=1, callbacks=[checkpoint])
