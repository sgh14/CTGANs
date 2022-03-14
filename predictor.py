from ctlearn.run_model import run_model
from tensorflow.keras import models
import yaml
import os


def train_predictor(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    run_model(config, mode='train')


def get_predictor(config_path, models_dir='models', model_name='predictor_lite'):
    os.makedirs(models_dir, exist_ok=True)
    predictor_path = os.path.join(models_dir, model_name)
    if os.path.relpath(predictor_path, models_dir) not in os.listdir(models_dir):
        train_predictor(config_path)

    predictor = models.load_model(predictor_path)

    return predictor