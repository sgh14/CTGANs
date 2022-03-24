import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow_addons.layers import SpectralNormalization
import numpy as np


class TransposeConvBlock(layers.Layer):
    def __init__(
        self,
        filters,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="same",
        use_batchnorm=True,
        use_bias=False,
        use_dropout=False,
        drop_value=0.3,
        kernel_initializer='orthogonal',
        spectral_norm=True
    ):
        super(TransposeConvBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.conv_transpose = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        if spectral_norm:
            self.conv_transpose = SpectralNormalization(self.conv_transpose)

        self.batch_normalization = layers.BatchNormalization()
        self.activation = activation
        self.dropout = layers.Dropout(drop_value)
        

    def call(self, inputs, training=False):
        x = self.conv_transpose(inputs)
        if self.use_batchnorm:
            x = self.batch_normalization(inputs=x, training=training)
        if self.activation:
            x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(inputs=x, training=training)
        
        return x


class UpsampleBlock(layers.Layer):
    def __init__(
        self,
        filters,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_batchnorm=True,
        use_bias=False,
        use_dropout=False,
        drop_value=0.3,
        kernel_initializer='orthogonal',
        spectral_norm=True
    ):
        super(UpsampleBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.upsampling = layers.UpSampling2D(up_size)
        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        if spectral_norm:
            self.conv = SpectralNormalization(self.conv)

        self.batch_normalization = layers.BatchNormalization()
        self.activation = activation
        self.dropout = layers.Dropout(drop_value)
        

    def call(self, inputs, training=False):
        x = self.upsampling(inputs)
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batch_normalization(inputs=x, training=training)
        if self.activation:
            x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(inputs=x, training=training)
        
        return x


class Generator(keras.Model):
    def __init__(self, g_config):
        super(Generator, self).__init__(name='generator')
        spectral_norm = g_config['spectral_normalization']
        self.latent_dim = g_config['latent_dim']
        self.dense = layers.Dense(**g_config['layers']['dense'])
        if spectral_norm:
            self.dense = SpectralNormalization(self.dense)

        self.batch_normalization = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(0.2)
        self.reshape = layers.Reshape(**g_config['layers']['reshape'])
        self.upsample_blocks = []
        for block_config in g_config['layers']['upsample_blocks']:
            if g_config['upsampling']:
                block = UpsampleBlock(**block_config, spectral_norm=spectral_norm)
            else:
                block = TransposeConvBlock(**block_config, spectral_norm=spectral_norm)

            self.upsample_blocks.append(block)

        #layers.Activation("tanh"))
        self.cropping = layers.Cropping2D(**g_config['layers']['cropping'])


    def _get_generator_inputs(self, labels):
        # Sample random points in the latent space and concatenate the labels.
        batch_size = tf.shape(list(labels.values())[0])[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        g_inputs = random_latent_vectors
        for task in labels.values():
            g_inputs = tf.concat([g_inputs, task], axis=1)

        return g_inputs
    

    def call(self, labels, training=False):
        g_inputs = self._get_generator_inputs(labels)
        x = self.dense(g_inputs)
        x = self.batch_normalization(inputs=x, training=training)
        x = self.activation(x)
        x = self.reshape(x)
        for block in self.upsample_blocks:
            x = block(inputs=x, training=training)
        
        x = self.cropping(x)

        return x


# TODO: implement a custom class that inherits from losses.Loss
# See if this allows to implement gradient penalty inside the loss class
def get_generator_loss(loss_name='basic', weights=np.array([1, 1, 1, 1])):
    if loss_name == 'basic':
        def generator_loss(d_outputs, p_outputs, d_labels, p_labels):
            # use from_logits=True to avoid using sigmoid activation when defining the discriminator
            bce = losses.BinaryCrossentropy(from_logits=True)
            cce = losses.CategoricalCrossentropy()
            mse = losses.MeanSquaredError()
            loss = weights[0]*bce(d_labels, d_outputs)
            if 'particletype' in p_labels.keys():
                loss += weights[1]*cce(p_labels['particletype'], p_outputs['particletype'])

            if 'energy' in p_labels.keys():
                loss += weights[2]*mse(p_labels['energy'], p_outputs['energy'])

            if 'direction' in p_labels.keys():
                loss += weights[3]*mse(p_labels['direction'], p_outputs['direction'])
                     
            return loss

    if loss_name == 'least_squares':
        def generator_loss(d_outputs, p_outputs, d_labels, p_labels):
            mse = losses.MeanSquaredError()
            cce = losses.CategoricalCrossentropy()
            loss = weights[0]*mse(d_labels, d_outputs)
            if 'particletype' in p_labels.keys():
                loss += weights[1]*cce(p_labels['particletype'], p_outputs['particletype'])

            if 'energy' in p_labels.keys():
                loss += weights[2]*mse(p_labels['energy'], p_outputs['energy'])

            if 'direction' in p_labels.keys():
                loss += weights[3]*mse(p_labels['direction'], p_outputs['direction'])
                     
            return loss

    if loss_name == 'w_gp':
        def generator_loss(d_outputs, p_outputs, d_labels, p_labels):
            cce = losses.CategoricalCrossentropy()
            mse = losses.MeanSquaredError()
            loss = weights[0]*(-tf.reduce_mean(d_outputs))
            if 'particletype' in p_labels.keys():
                loss += weights[1]*cce(p_labels['particletype'], p_outputs['particletype'])

            if 'energy' in p_labels.keys():
                loss += weights[2]*mse(p_labels['energy'], p_outputs['energy'])

            if 'direction' in p_labels.keys():
                loss += weights[3]*mse(p_labels['direction'], p_outputs['direction'])
                     
            return loss

    return generator_loss   