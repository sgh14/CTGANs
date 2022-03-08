import yaml
from dl1_data_handler.reader import DL1DataReaderSTAGE1, DL1DataReaderDL1DH
from ctlearn.data_loader import KerasBatchGenerator
from ctlearn.utils import *


def load_data(config_path, mode, batch_size=64, shuffle=False):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Set up the DL1DataReader
    config['Data'], data_format = setup_DL1DataReader(config, mode)
    # Create data reader
    if data_format == 'stage1':
        reader = DL1DataReaderSTAGE1(**config['Data'])
    elif data_format == 'dl1dh':
        reader = DL1DataReaderDL1DH(**config['Data'])

    # Set up the KerasBatchGenerator
    indices = list(range(len(reader)))
    dataset = KerasBatchGenerator(reader, indices, batch_size=batch_size, mode=mode, shuffle=shuffle)
    # Create a dictionary with useful dataset features
    features, labels = dataset.__getitem__(0)
    labels_dim = 0
    for task in labels.values():
        label_shape = task.shape[1] if np.ndim(task) == 2 else 1
        labels_dim += label_shape

    data_features = {
        'image_shape': dataset.img_shape,
        'batch_size': dataset.batch_size,
        'labels_dim': labels_dim
    }

    return dataset, data_features