import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
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
        drop_value=0.3
    ):
        super(TransposeConvBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.conv_transpose = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias
        )
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
        drop_value=0.3
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
            use_bias=use_bias
        )
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
    def __init__(self, latent_dim, upsampling=True):
        super(Generator, self).__init__(name='generator')
        self.latent_dim = latent_dim
        self.dense = layers.Dense(3 * 3 * 256, use_bias=False)
        self.batch_normalization = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(0.2)
        self.reshape = layers.Reshape((3, 3, 256))
        self.upsample_block1 = UpsampleBlock(128) if upsampling else TransposeConvBlock(128)
        self.upsample_block2 = UpsampleBlock(64) if upsampling else TransposeConvBlock(64)
        self.upsample_block3 = UpsampleBlock(32) if upsampling else TransposeConvBlock(32)
        self.upsample_block4 = UpsampleBlock(1) if upsampling else TransposeConvBlock(1) #layers.Activation("tanh"))
        self.cropping = layers.Cropping2D(((2, 3), (2, 3)))


    def _get_generator_inputs(self, labels):
        # TODO: allow labels to be a dict of arrays with shapes (batch, label_shape)
        # and also (label_shape) for single generation
        # Sample random points in the latent space and concatenate the labels.
        batch_size = tf.shape(list(labels.values())[0])[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        g_inputs = random_latent_vectors
        for task in labels.values():
            g_inputs = tf.concat([g_inputs, task], axis=1)

        return g_inputs
    

    def call(self, labels, training=False):
        x = self.dense(self._get_generator_inputs(labels))
        x = self.batch_normalization(inputs=x, training=training)
        x = self.activation(x)
        x = self.reshape(x)
        # Upsample to (6, 6, 128)
        x = self.upsample_block1(inputs=x, training=training)
        # Upsample to (12, 12, 64)
        x = self.upsample_block2(inputs=x, training=training)
        # Upsample to (24, 24, 32)
        x = self.upsample_block3(inputs=x, training=training)
        # Upsample to (48, 48, 1)
        x = self.upsample_block4(inputs=x, training=training) 
        # Crop to (43, 43, 1)
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
            mae = losses.MeanAbsoluteError()
            loss = weights[0]*bce(d_labels, d_outputs)
            if 'particletype' in p_labels.keys():
                loss += weights[1]*cce(p_labels['particletype'], p_outputs['particletype'])

            if 'energy' in p_labels.keys():
                loss += weights[2]*mae(p_labels['energy'], p_outputs['energy'])

            if 'direction' in p_labels.keys():
                loss += weights[3]*mae(p_labels['direction'], p_outputs['direction'])
                     
            return loss

    if loss_name == 'least_squares':
        def generator_loss(d_outputs, p_outputs, d_labels, p_labels):
            mse = losses.MeanSquaredError()
            cce = losses.CategoricalCrossentropy()
            mae = losses.MeanAbsoluteError()
            loss = weights[0]*mse(d_labels, d_outputs)
            if 'particletype' in p_labels.keys():
                loss += weights[1]*cce(p_labels['particletype'], p_outputs['particletype'])

            if 'energy' in p_labels.keys():
                loss += weights[2]*mae(p_labels['energy'], p_outputs['energy'])

            if 'direction' in p_labels.keys():
                loss += weights[3]*mae(p_labels['direction'], p_outputs['direction'])
                     
            return loss

    if loss_name == 'w_gp':
        def generator_loss(d_outputs, p_outputs, d_labels, p_labels):
            cce = losses.CategoricalCrossentropy()
            mae = losses.MeanAbsoluteError()
            loss = weights[0]*(-tf.reduce_mean(d_outputs))
            if 'particletype' in p_labels.keys():
                loss += weights[1]*cce(p_labels['particletype'], p_outputs['particletype'])

            if 'energy' in p_labels.keys():
                loss += weights[2]*mae(p_labels['energy'], p_outputs['energy'])

            if 'direction' in p_labels.keys():
                loss += weights[3]*mae(p_labels['direction'], p_outputs['direction'])
                     
            return loss

    return generator_loss   