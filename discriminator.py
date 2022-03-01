from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses


def conv_block(
    x,
    filters,
    activation,
    kernel_size=(5, 5),
    strides=(2, 2),
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.3,
    maxpool=False
):
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
    if maxpool:
        x = layers.MaxPool2D(pool_size=(2, 2))

    return x


def get_discriminator(input_shape):
    img_input = layers.Input(shape=input_shape)
    # x = layers.ZeroPadding2D((1, 1))(img_input)
    x = conv_block(img_input, 64, layers.LeakyReLU(0.2))
    # x = layers.ZeroPadding2D((1, 1))(x)
    x = conv_block(x, 128, layers.LeakyReLU(0.2), use_dropout=True)
    # x = layers.ZeroPadding2D((1, 1))(x)
    x = conv_block(x, 256, layers.LeakyReLU(0.2), use_dropout=True)
    x = conv_block(x, 512, layers.LeakyReLU(0.2))
    # x = conv_block(x, 1024, layers.LeakyReLU(0.2))
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    discriminator = models.Model(img_input, x, name="discriminator")

    return discriminator


def get_discriminator_loss(loss_name='basic'):
    if loss_name == 'basic':
        def discriminator_loss(labels, logits):
            bce = losses.BinaryCrossentropy(from_logits=True)
            loss = bce(labels, logits)

            return loss

    return discriminator_loss
