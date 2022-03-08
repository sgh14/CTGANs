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
config_files_dir = 'config_files'
GANs_config = os.path.join(config_files_dir, 'GANs.yml')
dataset, data_features = load_data(GANs_config, 'train', batch_size=64, shuffle=False)
latent_dim = 512

#%% TRAIN THE CTLearn MODEL IF THERE ISN'T ANY ALREADY SAVED
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
predictor_path = os.path.join(models_dir, 'predictor')
if os.path.relpath(predictor_path, models_dir) not in os.listdir(models_dir):
    train_predictor(os.path.join(config_files_dir, 'predictor.yml'))

predictor = models.load_model(predictor_path)

#%% BUILD THE GENERATOR
generator_in_shape = (latent_dim + data_features['labels_dim'],)
generator = get_generator(generator_in_shape)
generator.summary()
models_plots_dir = 'models_diagrams'
os.makedirs(models_plots_dir, exist_ok=True)
gen_plot_file = os.path.join(models_plots_dir, 'generator.png')
tf.keras.utils.plot_model(generator, show_shapes=True, to_file=gen_plot_file)

#%% BUILD THE DISCRIMINATOR
discriminator_in_shape = data_features['image_shape']
discriminator = get_discriminator(discriminator_in_shape)
discriminator.summary()
disc_plot_file = os.path.join(models_plots_dir, 'discriminator.png')
tf.keras.utils.plot_model(discriminator, show_shapes=True, to_file=disc_plot_file)

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

# TODO: add tensorboard callback
# Compile the GANs model.
gans.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)

#%% TRAIN GANS
# TODO: history = gans.fit() --> plot g_loss and d_loss evolution
gans.fit(dataset, epochs=1, verbose=1)

#%% SAVE THE MODELS
generator_path = os.path.join(models_dir, 'generator.h5')
discriminator_path = os.path.join(models_dir, 'discriminator.h5')
generator.save(generator_path)
discriminator.save(discriminator_path)

#%% GENERATE DATA
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)
features, labels = dataset.__getitem__(0)
images = generate_data(latent_dim, labels)
plot_grid(features['images'], save_path=os.path.join(images_dir, 'real_images.png'))
plot_grid(images, save_path=os.path.join(images_dir, 'generated_images.png'))
# %%