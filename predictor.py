from ctlearn.run_model import run_model
import yaml


def train_predictor(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    run_model(config, mode='train')