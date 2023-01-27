# Credits https://github.com/guilherme-pombo/keras_resnext_fpn
# Code written and expanded by Dmytro Zabolotnii, 2020/2021

"""
Implementation of Resnext FPN modified to incorporate multiple inputs and outputs of stage 1
Convolutional Recurrent Network for Road Boundary Extraction by Jiang L. et al
Version using strides convolutional layers
Running will generate model with standard configuration and save its description as txt
"""

from keras.models import Model
from keras.layers import Lambda, Activation, Conv2D, Conv2DTranspose, \
    Add, Input, BatchNormalization, UpSampling2D, Concatenate
from keras.layers.merge import concatenate, add
from keras.regularizers import l2

UP_DOWN_PYRAMID_SIZE = 128  # Depth for the ascending and descending portion of the pyramid


def resnext_fpn(input_shape, depth=(3, 3, 3, 3), cardinality=32, reg_factor=5e-4, batch_norm=True, batch_momentum=0.9,
                classification=False):
    """
    Resnext-50 is defined by (3, 4, 6, 3)
    Resnext-101 is defined by (3, 4, 23, 3)
    Resnext-152 is defined by (3, 8, 23, 3)
    Here we use less depth so we can use more filters in our convolutional layers due to memory restrictions
    :param input_shape: shape of the tensor input
    :param depth: amount of the layers on every block of the ascending portion of the pyramid
    :param cardinality: cardinality amount for ResNext block
    :param reg_factor: Regularization factor
    :param batch_norm: Batch normalization factor
    :param batch_momentum: Batch normalization momentum
    :param classification: whether to add final sigmoid layer for classification
    :return: full model
    """
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)

    # Pyramid up
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(reg_factor))(input_tensor)
    if batch_norm:
        x = BatchNormalization(axis=3, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    stage_1 = x

    # filters are cardinality * width * 2 for each depth level
    for i in range(depth[0]):
        x = bottleneck_block(x, 128, cardinality, strides=1, reg_factor=reg_factor, batch_norm=batch_norm)
    stage_2 = x

    # this can be done with a for loop but is more explicit this way
    x = bottleneck_block(x, 256, cardinality, strides=2, reg_factor=reg_factor, batch_norm=batch_norm)
    for idx in range(1, depth[1]):
        x = bottleneck_block(x, 256, cardinality, strides=1, reg_factor=reg_factor, batch_norm=batch_norm)
    stage_3 = x

    x = bottleneck_block(x, 512, cardinality, strides=2, reg_factor=reg_factor, batch_norm=batch_norm)
    for idx in range(1, depth[2]):
        x = bottleneck_block(x, 512, cardinality, strides=1, reg_factor=reg_factor, batch_norm=batch_norm)
    stage_4 = x

    x = bottleneck_block(x, 1024, cardinality, strides=2, reg_factor=reg_factor, batch_norm=batch_norm)
    for idx in range(1, depth[3]):
        x = bottleneck_block(x, 1024, cardinality, strides=1, reg_factor=reg_factor, batch_norm=batch_norm)
    stage_5 = x

    # Pyramid down
    P5 = Conv2D(UP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5', padding='same', kernel_regularizer=l2(reg_factor))(stage_5)
    if batch_norm:
        P5 = BatchNormalization(axis=3)(P5)
    P5 = Activation('relu')(P5)

    P4 = Add(name="fpn_p4add")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                Conv2D(UP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4', padding='same', kernel_regularizer=l2(reg_factor))(stage_4)])
    if batch_norm:
        P4 = BatchNormalization(axis=3)(P4)
    P4 = Activation('relu')(P4)

    P3 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                Conv2D(UP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3', padding='same', kernel_regularizer=l2(reg_factor))(stage_3)])
    if batch_norm:
        P3 = BatchNormalization(axis=3)(P3)
    P3 = Activation('relu')(P3)

    P2 = Add(name="fpn_p2add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                Conv2D(UP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2', padding='same', kernel_regularizer=l2(reg_factor))(stage_2)])
    if batch_norm:
        P2 = BatchNormalization(axis=3)(P2)
    P2 = Activation('relu')(P2)

    # HEAD GENERATION PART
    head1 = Conv2D(UP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2", kernel_regularizer=l2(reg_factor))(P2)
    if batch_norm:
        head1 = BatchNormalization(axis=3)(head1)
    head1 = Activation('relu')(head1)
    head1_1 = Conv2D(UP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head1_conv_1", kernel_regularizer=l2(reg_factor))(head1)

    head2 = Conv2D(UP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3", kernel_regularizer=l2(reg_factor))(P3)
    if batch_norm:
        head2 = BatchNormalization(axis=3)(head2)
    head2 = Activation('relu')(head2)
    head2_1 = Conv2D(UP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head2_conv_1", kernel_regularizer=l2(reg_factor))(head2)

    head3 = Conv2D(UP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4", kernel_regularizer=l2(reg_factor))(P4)
    if batch_norm:
        head3 = BatchNormalization(axis=3)(head3)
    head3 = Activation('relu')(head3)
    head3_1 = Conv2D(UP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head3_conv_1", kernel_regularizer=l2(reg_factor))(head3)

    head4 = Conv2D(UP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5", kernel_regularizer=l2(reg_factor))(P5)
    if batch_norm:
        head4 = BatchNormalization(axis=3)(head4)
    head4 = Activation('relu')(head4)
    head4_1 = Conv2D(UP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="head4_conv_1", kernel_regularizer=l2(reg_factor))(head4)

    # HEAD CONCAT PART
    f_p2_1 = UpSampling2D(size=(8, 8), name="pre_cat_2_1")(head4_1)
    f_p3_1 = UpSampling2D(size=(4, 4), name="pre_cat_3_1")(head3_1)
    f_p4_1 = UpSampling2D(size=(2, 2), name="pre_cat_4_1")(head2_1)
    f_p5_1 = head1_1

    x_1 = Concatenate(axis=3)([f_p2_1, f_p3_1, f_p4_1, f_p5_1])
    x_1 = Conv2D(UP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                 kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(x_1)

    # UPSAMPLING AND SMOOTHING PART
    x_1 = upsampling_block(x_1, result_filters=UP_DOWN_PYRAMID_SIZE // 2, reg_factor=reg_factor, batch_norm=batch_norm)
    res1 = Conv2D(1, (1, 1), padding="SAME", name="final_conv_1",
                  kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(x_1)
    if classification:
        res1 = Activation('sigmoid', name='final_activation_1')(res1)

    model = Model(input_tensor, res1)

    return model


def grouped_convolution_block(input, grouped_channels, cardinality, strides, reg_factor=5e-4, batch_norm=True):
    """
    ResNet/ResNext single convolutional layers
    :param input: input functional layer
    :param grouped_channels: amount of filters on the convolutional layer (is different from block when cardinality is not one)
    :param cardinality: amount of cardinality branches for ResNext
    :param strides: amount of strides used when downscaling
    :param reg_factor: regularization factor
    :param batch_norm: batch normalization bool
    :return: output functional layer
    """

    init = input
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(init)
        if batch_norm:
            x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(x)
        group_list.append(x)

    x = concatenate(group_list, axis=3)
    if batch_norm:
        x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x


def bottleneck_block(input, filters=64, cardinality=8, strides=1, reg_factor=5e-4, batch_norm=True):
    """
    ResNet/ResNext block of convolutional layers
    :param input: input functional layer
    :param filters: amount of filters on the block's convolutional layers
    :param cardinality: amount of cardinality for ResNext
    :param strides: amount of strides used when downscaling
    :param reg_factor: regularization factor
    :param batch_norm: batch normalization bool
    :return: output functional layer
    """
    init = input
    grouped_channels = int(filters / cardinality)

    if init._keras_shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(init)
        if batch_norm:
            init = BatchNormalization(axis=3)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(input)
    if batch_norm:
        x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = grouped_convolution_block(x, grouped_channels, cardinality, strides, reg_factor, batch_norm=batch_norm)
    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(reg_factor))(x)
    if batch_norm:
        x = BatchNormalization(axis=3)(x)

    x = add([init, x])
    x = Activation('relu')(x)
    return x


def upsampling_block(input, result_filters=UP_DOWN_PYRAMID_SIZE // 2, reg_factor=5e-4, batch_norm=True):
    """
    Bilinear upsampling followed by convolutional block
    :param input: input functional layer
    :param result_filters: amount of filters on the block's convolutional layers
    :param reg_factor: regularization factor
    :param batch_norm: batch normalization bool
    :return: output functional layer
    """
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(input)
    x = Conv2D(result_filters, (3, 3), padding="SAME",
               kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(x)
    if batch_norm:
        x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    return x


def upsampling_block_transpose(input, result_filters=UP_DOWN_PYRAMID_SIZE // 2, reg_factor=5e-4, batch_norm=True):
    """
    Upsampling achieved by using transposed convolutional block
    :param input: input functional layer
    :param result_filters: amount of filters on the block's convolutional layers
    :param reg_factor: regularization factor
    :param batch_norm: batch normalization bool
    :return: output functional layer
    """
    x = Conv2DTranspose(result_filters, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer='he_normal', kernel_regularizer=l2(reg_factor))(input)
    if batch_norm:
        x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    return x


if __name__ == '__main__':
    # Compiles standard configuration of model and saves it as txt

    INPUT_SHAPE = (256, 256, 2)

    model = resnext_fpn(INPUT_SHAPE, depth=(1, 1, 1, 1), cardinality=1)

    model.compile(optimizer='adam', loss='mean_squared_error')

    print(model.summary())
