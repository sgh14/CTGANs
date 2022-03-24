from ctlearn.run_model import run_model
from tensorflow.keras import models
import yaml
import os


def train_predictor(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    run_model(config, mode='train')

    return config['Logging']['model_directory']


def get_predictor(predefined_model_path, config_path):
    if predefined_model_path and os.path.exists(predefined_model_path):
        predictor = models.load_model(predefined_model_path)
    else:
        predictor_path = train_predictor(config_path)
        predictor = models.load_model(predictor_path)

    return predictor