from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
import numpy as np

def transpose_conv_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    use_bn=True,
    use_bias=False,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)

    return x


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
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias
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
    x = layers.Dense(3 * 3 * 256, use_bias=False)(z)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((3, 3, 256))(x)
    # Upsample to (6, 6, 128)
    x = upsample_block(x, 128, layers.LeakyReLU(0.2))
    # Upsample to (12, 12, 64)
    x = upsample_block(x, 64, layers.LeakyReLU(0.2))
    # Upsample to (24, 24, 32)
    x = upsample_block(x, 32, layers.LeakyReLU(0.2))
    # Upsample to (48, 48, 1)
    x = upsample_block(x, 1, layers.LeakyReLU(0.2)) #layers.Activation("tanh"))
    # Crop to (43, 43, 1)
    x = layers.Cropping2D(((2, 3), (2, 3)))(x)

    g_model = models.Model(z, x, name="generator")

    return g_model


def get_generator_loss(loss_name='basic', weights=np.array([1, 1, 1, 1])):
    if loss_name == 'basic':
        def generator_loss(disc_logits, pred_logits, disc_labels, pred_labels):
            bce = losses.BinaryCrossentropy(from_logits=True)
            cce = losses.CategoricalCrossentropy(from_logits=True)
            mae = losses.MeanAbsoluteError()
            loss = weights[0]*bce(disc_labels, disc_logits)
            if 'particletype' in pred_labels.keys():
                loss += weights[1]*cce(pred_labels['particletype'], pred_logits['particletype'])

            if 'energy' in pred_labels.keys():
                loss += weights[2]*mae(pred_labels['energy'], pred_logits['energy'])

            if 'direction' in pred_labels.keys():
                loss += weights[3]*mae(pred_labels['direction'], pred_logits['direction'])
                     
            return loss

    return generator_loss   