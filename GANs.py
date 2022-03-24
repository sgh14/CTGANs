import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import os
import shutil

from data_generator import plot_grid


class GANs(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        predictor,
        discriminator_extra_steps=1,
        generator_extra_steps=1,
        gp_weight=10
    ):
        super(GANs, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.predictor = predictor
        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps
        self.gp_weight = gp_weight
        

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GANs, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    
    def _gradient_penalty(self, batch_size, real_images, generated_images):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = generated_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp


    def _get_generator_loss_and_grads(self, batch_size, labels):
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(labels, training=True)
            # Get the discriminator and predictor logits for fake images
            d_outputs = self.discriminator(generated_images, training=False)
            p_outputs = self.predictor(generated_images)
            # Calculate the generator loss
            # disc_labels now are ones (real), since the generator expects to fool the discriminator
            g_loss = self.g_loss_fn(d_outputs, p_outputs, tf.ones((batch_size, 1)), labels)
        
        # Get the gradients w.r.t the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        return g_loss, g_gradient


    def _generator_train_step(self, real_images, labels):
        batch_size = tf.shape(real_images)[0]
        for _ in range(self.g_steps):
            g_loss, g_gradient = self._get_generator_loss_and_grads(batch_size, labels)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(g_gradient, self.generator.trainable_variables)
            )

        return g_loss


    def _get_discriminator_loss_and_grads(self, batch_size, real_images, generated_images):
        with tf.GradientTape() as tape:
            # Get the logits for the images
            d_outputs_on_real = self.discriminator(real_images, training=True)
            d_outputs_on_generated = self.discriminator(generated_images, training=True)
            # Calculate the discriminator loss using the fake and real image logits
            d_loss = self.d_loss_fn(tf.ones((batch_size, 1)), d_outputs_on_real)\
                   + self.d_loss_fn(tf.zeros((batch_size, 1)), d_outputs_on_generated)
            if self.gp_weight:
                # Calculate the gradient penalty
                gp = self._gradient_penalty(batch_size, real_images, generated_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss += gp * self.gp_weight                

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)

        return d_loss, d_gradient


    def _discriminator_train_step(self, real_images, labels):
        batch_size = tf.shape(real_images)[0]
        for _ in range(self.d_steps):
            # Generate fake images
            generated_images = self.generator(labels, training=False)
            # Get discriminator loss and gradients
            d_loss, d_gradient = self._get_discriminator_loss_and_grads(batch_size, real_images, generated_images)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        
        return d_loss


    def train_step(self, data):
        # Unpack the data.
        features, labels = data
        # TODO: this should be unnecessary
        real_images = tf.reshape(features['images'], (-1, 43, 43, 1)) #*2/self.max_intensity - 1
        for task in labels.keys():
            # TODO: remove 2 and task=='energy' to generalize.
            label_shape = 2 if task != 'energy' else 1
            # label_shape = tf.cond(tf.rank(labels[task])==2, lambda: tf.shape(labels[task])[-1], lambda: 1)
            # label_shape = tf.shape(labels[task])[1] if tf.rank(labels[task]) == tf.constant(2) else 1
            labels[task] = tf.reshape(labels[task], (-1, label_shape))

        # Discriminator train step
        d_loss = self._discriminator_train_step(real_images, labels)
        # Generator train step
        g_loss = self._generator_train_step(real_images, labels)
        
        return {"d_loss": d_loss, "g_loss": g_loss}


class Plot_and_save(callbacks.Callback):
    def __init__(self, dataset, epochs=1, nrows=5, ncols=5, images_dir='images', models_dir='models'):
        features, labels = dataset.__getitem__(0)
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
        self.logs = {'g_loss': [], 'd_loss': []}
    

    def _plot_loss(self, logs={}):
        # Plot g_loss and d_loss
        fig, ax = plt.subplots()
        ax.plot(logs['d_loss'], c='blue', label='d_loss')
        ax.plot(logs['g_loss'], c='red', label='g_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.savefig(os.path.join(self.images_dir, 'losses.png'))


    def _generate_and_save(self, epoch):
        # Save the generator and the discriminator
        self.model.generator.save(os.path.join(self.models_dir, f'generator_{epoch}'))
        self.model.discriminator.save(os.path.join(self.models_dir, f'discriminator_{epoch}'))
        # Plot a grid of generated images
        images = self.model.generator(self.labels).numpy()
        plot_grid(images, self.nrows, self.ncols, os.path.join(self.images_dir, f'generated_images_{epoch}'))


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
        # TODO: avoid repetition if the total_epochs%epochs == 0
        for key in ('g_loss', 'd_loss'):
            self.logs[key].append(logs[key])

        self._plot_loss(self.logs)
        epoch = '' # TODO: self.params.epochs?
        self._generate_and_save(epoch)
        # Plot a grid of real images corresponding to the same labels
        plot_grid(self.images, self.nrows, self.ncols, save_path=os.path.join(self.images_dir, 'real_images.png'))

