from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
import numpy as np


def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    up_size=(2, 2),
    padding="same",
    use_bn=True,
    use_bias=False,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)

    return x


def get_generator(input_shape):
    z = layers.Input(shape=input_shape)
    x = layers.Dense(4 * 4 * 256, use_bias=False)(z)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((4, 4, 256))(x)
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(0.2),
    )
    x = upsample_block(
        x,
        64,
        layers.LeakyReLU(0.2),
    )
    x = upsample_block(
        x,
        32,
        layers.LeakyReLU(0.2),
    )
    x = layers.Cropping2D((1, 1))(x)
    x = upsample_block(
        x,
        16,
        layers.LeakyReLU(0.2),
    )
    x = layers.Cropping2D((1, 1))(x)
    x = upsample_block(
        x,
        1, 
        layers.Activation("tanh"), 
    )
    x = layers.Cropping2D((1, 1))(x)

    g_model = models.Model(z, x, name="generator")

    return g_model


def get_generator_loss(loss_name='normal'):
    if loss_name == 'normal':
        def generator_loss(disc_logits, pred_logits, disc_labels, pred_labels):
            bce = losses.BinaryCrossentropy(from_logits=True)
            cce = losses.CategoricalCrossentropy(from_logits=True)
            mae = losses.MeanAbsoluteError()
            w = np.array([1, 0, 0, 0])
            # TODO: use weights and allow tunning
            loss = w[0]*bce(disc_labels, disc_logits)
            if 'particletype' in pred_labels.keys():
                loss += w[1]*cce(pred_labels['particletype'], pred_logits['particletype'])

            if 'energy' in pred_labels.keys():
                loss += w[2]*mae(pred_labels['energy'], pred_logits['energy'])

            if 'direction' in pred_labels.keys():
                loss += w[3]*mae(pred_labels['direction'], pred_logits['direction'])
                     
            return loss

    return generator_loss   