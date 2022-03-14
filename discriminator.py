from tensorflow import keras
from tensorflow.keras import layers, losses


class ConvBlock(layers.Layer):
    def __init__(
        self,
        filters,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_batchnorm=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3
    ):
        super(ConvBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias
        )
        # TODO: make sure batchnorm and dropout don't appear in model_plot when use_bn and use_drop = False
        self.batch_normalization = layers.BatchNormalization()
        self.activation = activation
        self.dropout = layers.Dropout(drop_value)
        

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.use_batchnorm:
            x = self.batch_normalization(inputs=x, training=training)
        if self.activation:
            x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(inputs=x, training=training)
        
        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__(name='discriminator')
        self.zeropadding = layers.ZeroPadding2D(((2, 3), (2, 3)))
        self.conv_block1 = ConvBlock(64)
        self.conv_block2 = ConvBlock(128, use_dropout=True)
        self.conv_block3 = ConvBlock(256, use_dropout=True)
        self.conv_block4 = ConvBlock(512)
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.2)        
        self.dense = layers.Dense(1) # default activation is linear
    

    def call(self, inputs, training=False):
        x = self.zeropadding(inputs)
        # Downsample to (24, 24, 64)
        x = self.conv_block1(inputs=x, training=training)
        # Downsample to (12, 12, 128)
        x = self.conv_block2(inputs=x, training=training)
        # Downsample to (6, 6, 256)
        x = self.conv_block3(inputs=x, training=training)
        # Downsample to (3, 3, 512)
        x = self.conv_block4(inputs=x, training=training)
        x = self.flatten(x)
        x = self.dropout(inputs=x, training=training)
        x = self.dense(x)

        return x


def get_discriminator_loss(loss_name='basic'):
    if loss_name == 'basic':
        def discriminator_loss(labels, logits):
            # use from_logits=True to avoid using sigmoid activation when defining the discriminator
            bce = losses.BinaryCrossentropy(from_logits=True)
            loss = bce(labels, logits)

            return loss

    return discriminator_loss
