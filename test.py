#%%
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import time
import json
import yaml
import os

from data_loader import load_data


config_path = 'config_files/test-gamma.yml' # input('Configuration file path: ')
with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

predictions_dir = os.path.join(config['results_dir'], 'predictions')
os.makedirs(predictions_dir, exist_ok=True)

#%%
# Load the dataset
test_set = load_data(config, **config['Input'])
# Load the prediction models
particle_type_classifier = models.load_model(config['particle_type_classifier'])
energy_regressor = models.load_model(config['energy_regressor'])
direction_regressor = models.load_model(config['direction_regressor'])
# Create empty lists to store the predictions
particle_type_predictions_on_real = []
energy_predictions_on_real = []
direction_predictions_on_real = []

# Get errors for real data
labels = {
    'particletype': [],
    'energy': [],
    'direction': []
}
for batch in range(test_set.__len__()):
    # Load the real batch
    real_imgs, batch_labels = test_set.__getitem__(batch)
    for key, value in batch_labels.items():
        labels[key].append(value)

    # Predict particle type
    particle_type_predictions_on_real.append(particle_type_classifier.predict(real_imgs))
    # Get error in energy regression
    energy_predictions_on_real.append(energy_regressor.predict(real_imgs))
    # Get error in direction regression
    direction_predictions_on_real.append(direction_regressor.predict(real_imgs))

particle_type_predictions_on_real = np.concatenate(particle_type_predictions_on_real, axis=0)
energy_predictions_on_real = np.concatenate(energy_predictions_on_real, axis=0)
direction_predictions_on_real = np.concatenate(direction_predictions_on_real, axis=0)

np.savetxt(os.path.join(predictions_dir, 'particle_type_predictions_on_real.txt'), particle_type_predictions_on_real)
np.savetxt(os.path.join(predictions_dir, 'energy_predictions_on_real.txt'), energy_predictions_on_real)
np.savetxt(os.path.join(predictions_dir, 'direction_predictions_on_real.txt'), direction_predictions_on_real)

for key in labels.keys():
        labels[key] = np.concatenate(labels[key], axis=0).tolist()

with open(os.path.join(config['results_dir'], 'labels.json'), 'w') as file:
    json.dump(labels, file)

#%%
with open(os.path.join(config['results_dir'], 'labels.json'), 'r') as file:
    labels = json.load(file)

for key in labels.keys():
        labels[key] = np.array(labels[key])

# particle_type_auc_on_real = metrics.roc_auc_score(labels['particletype'], particle_type_predictions_on_real)
particle_type_acc_on_real = metrics.accuracy_score(np.argmax(labels['particletype'], axis=1), np.argmax(particle_type_predictions_on_real, axis=1))
energy_mae_on_real = metrics.mean_absolute_error(labels['energy'], energy_predictions_on_real)
direction_mae_on_real = metrics.mean_absolute_error(labels['direction'], direction_predictions_on_real)

# Get errors for generated data
generators = os.listdir(config['generators_dir'])
generators = [g_path for g_path in generators if g_path.startswith('generator_')]

#%%
particle_type_acc_on_generated = []
energy_mae_on_generated = []
direction_mae_on_generated = []
generation_time = []
# Predict for every checkpoint
for generator_path in generators[0:2]:
    i = generator_path.split('_')[-1]
    # Load the generator
    generator = models.load_model(os.path.join(config['generators_dir'], generator_path))
    # Generate images for the same labels and measure the time it takes
    labels_for_generation = {key: value for key, value in labels.items() if key!='particletype'}
    tic = time.time()
    generated_imgs = generator(labels_for_generation)
    toc = time.time()
    generation_time.append(toc-tic)
    # Predict particle type on real and generated images
    particle_type_predictions_on_generated = particle_type_classifier.predict(generated_imgs)
    # Get error in energy regression
    energy_predictions_on_generated = energy_regressor.predict(generated_imgs)
    # Get error in direction regression
    direction_predictions_on_generated = direction_regressor.predict(generated_imgs)

    np.savetxt(os.path.join(predictions_dir, f'particle_type_predictions_on_generated_{i}.txt'), particle_type_predictions_on_generated)
    np.savetxt(os.path.join(predictions_dir, f'energy_predictions_on_generated_{i}.txt'), energy_predictions_on_generated)
    np.savetxt(os.path.join(predictions_dir, f'direction_predictions_on_generated_{i}.txt'), direction_predictions_on_generated)

    particle_type_acc_on_generated.append(metrics.accuracy_score(np.argmax(labels['particletype'], axis=1), np.argmax(particle_type_predictions_on_generated, axis=1)))
    energy_mae_on_generated.append(metrics.mean_absolute_error(labels['energy'], energy_predictions_on_generated))
    direction_mae_on_generated.append(metrics.mean_absolute_error(labels['direction'], direction_predictions_on_generated))

particle_type_acc_on_generated = np.array(particle_type_acc_on_generated)
energy_mae_on_generated = np.array(energy_mae_on_generated)
direction_mae_on_generated = np.array(direction_mae_on_generated)

#%%
# TODO: create x vector with epochs
epochs = np.linspace(1, config['total_epochs'], config['epochs_between_models'])
# AUC plot
fig, ax = plt.subplots()
ax.plot(epochs, particle_type_auc_on_generated, label='AUC for generated data')
ax.hlines([particle_type_auc_on_real], 1, config['total_epochs'], ls='--', label='AUC for real data')
fig.savefig(os.path.join(config['results_dir'], 'AUC_plot.pdf'))
fig.savefig(os.path.join(config['results_dir'], 'AUC_plot.png'))

# TODO: ROC plot for best AUC

# Energy MAE plot
fig, ax = plt.subplots()
ax.plot(epochs, energy_mae_on_generated, label='Energy MAE for generated data')
ax.hlines([energy_mae_on_real], 1, config['total_epochs'], ls='--', label='Energy MAE for real data')
fig.savefig(os.path.join(config['results_dir'], 'energy_MAE_plot.pdf'))
fig.savefig(os.path.join(config['results_dir'], 'energy_MAE_plot.png'))

# Direction MAE plot
fig, ax = plt.subplots()
ax.plot(epochs, direction_mae_on_generated, label='Arrival direction MAE for generated data')
ax.hlines([direction_mae_on_real], 1, config['total_epochs'], ls='--', label='Arrival direction MAE for real data')
fig.savefig(os.path.join(config['results_dir'], 'direction_MAE_plot.pdf'))
fig.savefig(os.path.join(config['results_dir'], 'direction_MAE_plot.png'))

# def plot_classifier_values(
#     gamma_classifier_values,
#     proton_classifier_values,
#     output_filename='classifier_histogram'
# ):
#     # Make the plot
#     plt.figure()

#     # Plot the histograms for both classifier values
#     bins = np.linspace(0, 1, 100)
#     plt.hist(gamma_classifier_values, bins, alpha=0.5, label='Gamma')
#     plt.hist(proton_classifier_values, bins, alpha=0.5, label='Proton')

#     plt.xlabel('Classifier value')
#     plt.ylabel('Counts')
#     # plt.title('Histogram of classifier values')
#     plt.legend(loc='upper center')

#     plt.savefig(output_filename + '.pdf', bbox_inches='tight')
#     plt.savefig(output_filename + '.png', bbox_inches='tight')


# def plot_roc_curves(
#     gamma_classifier_values,
#     proton_classifier_values,
#     output_filename='roc_curves'
# ):
#     gamma_true_values = np.ones(len(gamma_classifier_values))
#     proton_true_values = np.zeros(len(proton_classifier_values))

#     # Make the plot
#     plt.figure()

#     # Plot the ROC curve
#     classifier_values = np.concatenate((gamma_classifier_values, proton_classifier_values))
#     true_values = np.concatenate((gamma_true_values, proton_true_values))

#     fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_values, classifier_values)
#     auc = sklearn.metrics.auc(fpr, tpr)

#     plt.plot(fpr, tpr, lw=2, label='AUC = {:.2f}'.format(auc))

#     # Finish the plot
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])

#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     # plt.title('Receiver Operating Characteristic')

#     plt.legend(loc='lower right')

#     plt.savefig(output_filename + '.pdf', bbox_inches='tight')
#     plt.savefig(output_filename + '.png', bbox_inches='tight')

'''
# Get errors for generated data
models = os.listdir(config['GANs']['models_dir'])
models = [model_path for model_path in models if model_path.startswith('generator_')]

particle_type_auc_on_generated = []
energy_mae_on_generated = []
direction_mae_on_generated = []
# Predict for every checkpoint
for model_path in models:
    # Load the generator
    generator = models.load_model(os.path.join(config['GANs']['models_dir'], model_path))
    # Create empty lists to store the predictions
    particle_type_predictions_on_generated = []
    energy_predictions_on_generated = []
    direction_predictions_on_generated = []
    # Create an empty list to store the time it takes to generate the data
    generation_time = []
    for batch in range(test_set.__len__()):
        # Load the real batch
        real_imgs, labels = test_set.__getitem__(batch)
        # Generate images for the same labels and measure the time it takes
        tic = time.clock()
        generated_imgs = generator(labels)
        toc = time.clock()
        generation_time.append(toc-tic)
        # Predict particle type on real and generated images
        particle_type_predictions_on_generated.append(particle_type_classifier.predict(generated_imgs))
        # Get error in energy regression
        energy_predictions_on_generated.append(energy_regressor.predict(generated_imgs))
        # Get error in direction regression
        direction_predictions_on_generated.append(direction_regressor.predict(generated_imgs))

    particle_type_predictions_on_generated = np.concatenate(particle_type_predictions_on_generated, axis=0).squeeze()
    energy_predictions_on_generated = np.concatenate(energy_predictions_on_generated, axis=0).squeeze()
    energy_predictions_on_generated = np.concatenate(particle_type_predictions_on_generated, axis=0).squeeze()
    generation_time = np.array(generation_time)

    # TODO: save predictions

    particle_type_auc.append()
    energy_mae.append()
    direction_mae.append()

particle_type_auc.append()
energy_mae.append()
direction_mae.append()
'''

# file_paths = {
#     'particletype': {
#         'real': 'test/particle_type_predictions_on_real.txt',
#         'generated': 'test/particle_type_predictions_on_generated.txt'
#     },
#     'energy': {
#         'real': 'test/energy_predictions_on_real.txt',
#         'generated': 'test/energy_predictions_on_generated.txt'
#     },
#     'direction': {
#         'real': 'test/direction_predictions_on_real.txt',
#         'generated': 'test/direction_predictions_on_generated.txt'
#     }
# }
# %%
