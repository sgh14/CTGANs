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
):
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)

    return x


def get_discriminator(input_shape):
    img_input = layers.Input(shape=input_shape)
    # Zero pad the input to make the input images size to (116, 116, 1).
    x = layers.ZeroPadding2D((1, 1))(img_input)
    x = conv_block(
        x,
        64,
        activation=layers.LeakyReLU(0.2),
    )
    x = layers.ZeroPadding2D((1, 1))(img_input)
    x = conv_block(
        x,
        128,
        activation=layers.LeakyReLU(0.2),
        use_dropout=True,
    )
    x = layers.ZeroPadding2D((1, 1))(img_input)
    x = conv_block(
        x,
        256,
        activation=layers.LeakyReLU(0.2),
        use_dropout=True,
    )
    x = conv_block(
        x,
        512,
        activation=layers.LeakyReLU(0.2),
        use_dropout=True
    )
    x = conv_block(
        x,
        1024,
        activation=layers.LeakyReLU(0.2),
        use_dropout=True
    )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    discriminator = models.Model(img_input, x, name="discriminator")

    return discriminator


def get_discriminator_loss(loss_name='normal'):
    if loss_name == 'normal':
        def discriminator_loss(labels, logits):
            bce = losses.BinaryCrossentropy(from_logits=True)
            loss = bce(labels, logits)

            return loss

    return discriminator_loss
