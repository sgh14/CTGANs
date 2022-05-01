#%%
from tensorflow.keras import models
import numpy as np
from sklearn import metrics
import time
import json
import yaml
import os

from data_loader import load_data

# Load the config file for testing
config_path = input('Configuration file path: ') # 'config_files/test-gamma.yml'
with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

# Create the directory to store the results
predictions_dir = os.path.join(config['results_dir'], 'predictions')
os.makedirs(predictions_dir, exist_ok=True)

# Load the dataset
test_set = load_data(config, **config['Input'])
# Load the (CTLearn) predictive models
particle_type_classifier = models.load_model(config['particle_type_classifier'])
energy_regressor = models.load_model(config['energy_regressor'])
direction_regressor = models.load_model(config['direction_regressor'])
# Create empty lists to store the predictions
particle_type_predictions_on_real = []
energy_predictions_on_real = []
direction_predictions_on_real = []
# Labels will be written to a json file for possible future usage
labels = {
    'particletype': [],
    'energy': [],
    'direction': []
}
# Get errors for real data
for batch in range(test_set.__len__()):
    # Load the real batch
    real_imgs, batch_labels = test_set.__getitem__(batch)
    # Save the labels in the dictionary
    for key, value in batch_labels.items():
        labels[key].append(value)

    # Predict particle type on real data
    particle_type_predictions_on_real.append(particle_type_classifier.predict(real_imgs))
    # Predict energy on real data 
    energy_predictions_on_real.append(energy_regressor.predict(real_imgs))
    # Predict arrival direction on real
    direction_predictions_on_real.append(direction_regressor.predict(real_imgs))

# Turn the stored predictions into numpy arrays (remove batch dimension)
particle_type_predictions_on_real = np.concatenate(particle_type_predictions_on_real, axis=0)
energy_predictions_on_real = np.concatenate(energy_predictions_on_real, axis=0)
direction_predictions_on_real = np.concatenate(direction_predictions_on_real, axis=0)
# Save the predictions as txt files
np.savetxt(os.path.join(predictions_dir, 'particle_type_predictions_on_real.txt'), particle_type_predictions_on_real)
np.savetxt(os.path.join(predictions_dir, 'energy_predictions_on_real.txt'), energy_predictions_on_real)
np.savetxt(os.path.join(predictions_dir, 'direction_predictions_on_real.txt'), direction_predictions_on_real)
# Remove batch dimension from label lists
for key in labels.keys():
        labels[key] = np.concatenate(labels[key], axis=0).tolist()

# Save the labels as a json file
with open(os.path.join(config['results_dir'], 'labels.json'), 'w') as file:
    json.dump(labels, file)

# with open(os.path.join(config['results_dir'], 'labels.json'), 'r') as file:
#     labels = json.load(file)

# Turn label lists into numpy arrays
for key in labels.keys():
        labels[key] = np.array(labels[key])

# Compute the metrics for real data
particle_type_acc_on_real = metrics.accuracy_score(np.argmax(labels['particletype'], axis=1), np.argmax(particle_type_predictions_on_real, axis=1))
energy_mae_on_real = metrics.mean_absolute_error(labels['energy'], energy_predictions_on_real)
direction_mae_on_real = metrics.mean_absolute_error(labels['direction'], direction_predictions_on_real)
# Save the metrics for real data
with open(os.path.join(config['results_dir'], 'metrics_on_real.json'), 'w') as file:
    json.dump({
        'particle_type_acc_on_real': particle_type_acc_on_real,
        'energy_mae_on_real': energy_mae_on_real,
        'direction_mae_on_real': direction_mae_on_real
    }, file)

# Create empty lists to store the metrics for generated data
particle_type_acc_on_generated = []
energy_mae_on_generated = []
direction_mae_on_generated = []
generation_time = []
w_distance = []
# Predict on generated data for every checkpoint of the models
for i in range(config['epochs_between_models'], config['total_epochs']+1, config['epochs_between_models']):
    print('Epoch:', i)
    # Load the generator and the discriminator
    generator_path = os.path.join(config['models_dir'], f'generator_{i}')
    discriminator_path = os.path.join(config['models_dir'], f'discriminator_{i}')
    generator = models.load_model(generator_path)
    discriminator = models.load_model(discriminator_path)
    # Create empty lists to store the predictions for generated data
    particle_type_predictions_on_generated = []
    energy_predictions_on_generated = []
    direction_predictions_on_generated = []
    generation_time_i = []
    w_distance_i = []
    for batch in range(test_set.__len__()):
        # Load the real batch
        real_imgs, batch_labels = test_set.__getitem__(batch)
        # Generate images for the same labels and measure the time it takes
        labels_for_generation = {key: value for key, value in batch_labels.items() if key!='particletype'}
        tic = time.time()
        generated_imgs = generator(labels_for_generation)
        toc = time.time()
        generation_time_i.append(toc-tic)
        # Predict particle type on generated images
        particle_type_predictions_on_generated.append(particle_type_classifier.predict(generated_imgs))
        # Predict energy on generated images
        energy_predictions_on_generated.append(energy_regressor.predict(generated_imgs))
        # Predict arrival direction on generated images
        direction_predictions_on_generated.append(direction_regressor.predict(generated_imgs))
        # Get W1-distance between real and generated data
        w_distance_i.append(np.mean(discriminator(generated_imgs))-np.mean(discriminator(real_imgs['images'])))

    # Turn the stored predictions into numpy arrays (remove batch dimension)
    particle_type_predictions_on_generated = np.concatenate(particle_type_predictions_on_generated, axis=0)
    energy_predictions_on_generated = np.concatenate(energy_predictions_on_generated, axis=0)
    direction_predictions_on_generated = np.concatenate(direction_predictions_on_generated, axis=0)
    generation_time.append(np.mean(generation_time_i))
    # Save the predictions as txt files
    np.savetxt(os.path.join(predictions_dir, f'particle_type_predictions_on_generated_{i}.txt'), particle_type_predictions_on_generated)
    np.savetxt(os.path.join(predictions_dir, f'energy_predictions_on_generated_{i}.txt'), energy_predictions_on_generated)
    np.savetxt(os.path.join(predictions_dir, f'direction_predictions_on_generated_{i}.txt'), direction_predictions_on_generated)
    # Append the metrics for the current checkpoint to the general lists
    particle_type_acc_on_generated.append(metrics.accuracy_score(np.argmax(labels['particletype'], axis=1), np.argmax(particle_type_predictions_on_generated, axis=1)))
    energy_mae_on_generated.append(metrics.mean_absolute_error(labels['energy'], energy_predictions_on_generated))
    direction_mae_on_generated.append(metrics.mean_absolute_error(labels['direction'], direction_predictions_on_generated))
    w_distance.append(np.mean(np.array(w_distance_i)))

# Turn the stored metrics (for all checkpoints) into numpy arrays
particle_type_acc_on_generated = np.array(particle_type_acc_on_generated)
energy_mae_on_generated = np.array(energy_mae_on_generated)
direction_mae_on_generated = np.array(direction_mae_on_generated)
generation_time = np.array(generation_time)
w_distance = np.array(w_distance)
# Save the metrics (for all checkpoints) as txt files
np.savetxt(os.path.join(config['results_dir'], 'particle_type_acc_on_generated.txt'), particle_type_acc_on_generated)
np.savetxt(os.path.join(config['results_dir'], 'energy_mae_on_generated.txt'), energy_mae_on_generated)
np.savetxt(os.path.join(config['results_dir'], 'direction_mae_on_generated.txt'), direction_mae_on_generated)
np.savetxt(os.path.join(config['results_dir'], 'generation_time.txt'), generation_time)
np.savetxt(os.path.join(config['results_dir'], 'w_distance.txt'), w_distance)
