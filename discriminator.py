from tensorflow.keras import layers, models, losses


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
    x = layers.ZeroPadding2D(((2, 3), (2, 3)))(img_input)
    # Downsample to (24, 24, 64)
    x = conv_block(x, 64, layers.LeakyReLU(0.2))
    # Downsample to (12, 12, 128)
    x = conv_block(x, 128, layers.LeakyReLU(0.2), use_dropout=True)
    # Downsample to (6, 6, 256)
    x = conv_block(x, 256, layers.LeakyReLU(0.2), use_dropout=True)
    # Downsample to (3, 3, 512)
    x = conv_block(x, 512, layers.LeakyReLU(0.2))
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x) # TODO: use linear activation when using W_GP loss

    discriminator = models.Model(img_input, x, name="discriminator")

    return discriminator


def get_discriminator_loss(loss_name='basic'):
    if loss_name == 'basic':
        def discriminator_loss(labels, logits):
            bce = losses.BinaryCrossentropy(from_logits=True)
            loss = bce(labels, logits)

            return loss

    return discriminator_loss
