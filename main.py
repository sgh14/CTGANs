#%%
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
import os

from data_loader import *
from predictor import *
from discriminator import get_discriminator, get_discriminator_loss
from generator import get_generator, get_generator_loss
from GANs import *
from data_generator import *

#%% LOAD DATA
# TODO: image normalization
config_path = 'LSTCam_TRN_lowcut.yml'
dataset, data_features = load_data(config_path, 'train', batch_size=64, shuffle=False)
latent_dim = 128

#%% TRAIN THE CTLearn MODEL IF THERE ISN'T ANY ALREADY SAVED
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
predictor_path = os.path.join(models_dir, 'predictor')
if os.path.relpath(predictor_path, models_dir) not in os.listdir(models_dir):
    train_predictor(config_path)

predictor = models.load_model(predictor_path)

#%% BUILD THE GENERATOR
generator_in_shape = (latent_dim + 2 + 1 + 2,)
generator = get_generator(generator_in_shape)
generator.summary()
tf.keras.utils.plot_model(generator, show_shapes=True)

#%% BUILD THE DISCRIMINATOR
discriminator_in_shape = data_features['image_shape']
discriminator = get_discriminator(discriminator_in_shape)
discriminator.summary()
tf.keras.utils.plot_model(discriminator, show_shapes=True)

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

# TODO: añadir tensorboard callback
# Compile the GANs model.
gans.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)

#%% TRAIN GANS
gans.fit(dataset, epochs=10, verbose=1)

#%% SAVE THE MODELS
generator_path = os.path.join(models_dir, 'generator.h5')
discriminator_path = os.path.join(models_dir, 'discriminator.h5')
generator.save(generator_path)
discriminator.save(discriminator_path)

#%% GENERATE DATA
features, labels = dataset.__getitem__(0)
images = generate_data(latent_dim, labels)
plot_image(images[0])
# predictor(images[0].numpy().reshape(-1, 114, 114, 1))
# %%