from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import shutil
import json

from data_generator import plot_grid


class Checkpoint(callbacks.Callback):
    def __init__(
        self,
        dataset,
        epochs=1,
        nrows=5,
        ncols=5,
        images_dir='images',
        models_dir='models',
        initial_epoch=0
    ):
        features, labels = dataset.__getitem__(0)
        if initial_epoch == 0:
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            for file in os.listdir(models_dir):
                if file.startswith('generator_') or file.startswith('discriminator_'):
                    shutil.rmtree(os.path.join(models_dir, file))

            for file in os.listdir(images_dir):
                if file.startswith('generated_images_'):
                    os.remove(os.path.join(images_dir, file))

        self.labels = labels
        self.images = features['images']
        self.epochs = epochs
        self.nrows = nrows
        self.ncols = ncols
        self.images_dir = images_dir
        self.models_dir = models_dir
        self.initial_epoch = initial_epoch
        self.loss_file_path = os.path.join(self.models_dir, 'losses.json')
        
        if initial_epoch != 0 and os.path.exists(self.loss_file_path):
            with open(self.loss_file_path, 'r') as loss_file:
                self.logs = json.load(loss_file)
        else:
            self.logs = {'g_loss': [], 'd_loss': []}
    

    def _plot_loss(self, logs={}):
        # Plot g_loss and d_loss
        fig, ax1 = plt.subplots()
        epochs = [epoch+1 for epoch in range(len(logs['d_loss']))]
        ax1.plot(epochs, logs['d_loss'], c='blue', label='$D$ loss')
        ax2 = ax1.twinx()
        ax2.plot(epochs, logs['g_loss'], c='red', label='$G$ loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Discriminator loss')
        ax2.set_ylabel('Generator loss')
        fig.legend(loc='upper center', ncol=2)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(-1,1), useMathText=True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(os.path.join(self.images_dir, 'losses.pdf'))
        fig.savefig(os.path.join(self.images_dir, 'losses.png'))


    def _generate_and_save(self, epoch):
        epoch += self.initial_epoch
        # Save the generator and the discriminator
        self.model.generator.save(os.path.join(self.models_dir, f'generator_{epoch+1}'))
        self.model.discriminator.save(os.path.join(self.models_dir, f'discriminator_{epoch+1}'))
        # Plot a grid of generated images
        images = self.model.generator(self.labels).numpy()
        plot_grid(images, self.nrows, self.ncols, os.path.join(self.images_dir, f'generated_images_{epoch+1}'))


    # def on_train_begin(self, logs=None):
    #     # Print generator summary and plot the model
    #     self.model.generator.summary()
    #     gen_plot_file = os.path.join(self.images_dir, 'generator.png')
    #     tf.keras.utils.plot_model(self.model.generator, show_shapes=True, to_file=gen_plot_file)
    #     # Print discriminator summary and plot the model
    #     self.model.discriminator.summary()
    #     disc_plot_file = os.path.join(self.images_dir, 'discriminator.png')
    #     tf.keras.utils.plot_model(self.model.discriminator, show_shapes=True, to_file=disc_plot_file)


    # def on_batch_end(self, batch, logs=None):
    #     for key in ('g_loss', 'd_loss'):
    #         self.logs[key].append(logs[key])


    def on_epoch_end(self, epoch, logs=None):
        for key in ('g_loss', 'd_loss'):
            self.logs[key].append(logs[key])
        
        self._plot_loss(self.logs)

        if epoch%self.epochs == 0:
            self._generate_and_save(epoch)


    def on_train_end(self, logs=None):
        with open(os.path.join(self.models_dir, 'losses.json'), 'w') as file:
            json.dump(self.logs, file)

        # Plot a grid of real images corresponding to the same labels
        plot_grid(self.images, self.nrows, self.ncols, save_path=os.path.join(self.images_dir, 'real_images.png'))

