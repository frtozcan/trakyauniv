# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 00:01:07 2022

@author: FO_KLU
"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, \
    BatchNormalization, Dropout, Lambda, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate

def conv2d_bn(x,filters,num_row,num_col,padding="same",strides=(1,1)):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale="False")(x)
    x = Activation("relu")(x)
    return x

def inc_block_a(x):
    branch1x1= conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    branch_pool = MaxPool2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x= concatenate([branch1x1, branch5x5,branch3x3,branch_pool], axis=3)
    return x

def reduction_block_a(x):
    branch3x3 = conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding="same")

    branch3x3bdl = conv2d_bn(x, 16, 1, 1)
    branch3x3bdl = conv2d_bn(branch3x3bdl, 32, 3, 3)
    branch3x3bdl = conv2d_bn(branch3x3bdl, 32, 3, 3, strides=(2, 2), padding="same",)

    branch_pool = MaxPool2D((3, 3), strides=(1, 1), padding="same")(x)
    x1 = Concatenate([branch3x3, branch3x3bdl])
    x2 = Concatenate([x1, branch_pool])
    return x2





def decoder_block(input, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def decoder_block_con(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)  # Not in the original network.
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  # Not in the original network
    x = Activation("relu")(x)

    return x


# Encoder block: Conv block followed by maxpooling
def yan_conv_block(x, num_filters, dropout=0.5, batch_norm=True):
    x = conv_block(x, num_filters)
    if batch_norm is True:
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    if dropout > 0:
        x = Dropout(dropout)(x)
    x = MaxPool2D((2, 2))(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


# Decoder block
# skip features gets input from encoder for concatenation

# Build Unet using the blocks
def Aim_Unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s1_1 = inc_block_a(s1)
    s1_1 = inc_block_a(s1_1)
    #s1_1 = inc_block_a(s1_1)
    #s1_1= reduction_block_a(s1_1)
    s1_1= conv_block(s1_1, 64)
    s1=Concatenate()([s1,s1_1])

    s2, p2 = encoder_block(p1, 128)
    s2_1 = inc_block_a(s2)
    s2_1 = inc_block_a(s2_1)
    #s2_1 = inc_block_a(s2_1)

    #s2_1 = reduction_block_a(s2_1)
    s2_1 = conv_block(s2_1, 128)
    s2=Concatenate()([s2,s2_1])



    s3, p3 = encoder_block(p2, 256)
    s3_1 = inc_block_a(s3)
    s3_1 = inc_block_a(s3_1)
    #s3_1 = inc_block_a(s3_1)

    #s3_1 = reduction_block_a(s3_1)
    s3_1 = conv_block(s3_1, 256)
    s3=Concatenate()([s3,s3_1])


    s4, p4 = encoder_block(p3, 512)
    s4_1 = inc_block_a(s4)
    s4_1 = inc_block_a(s4_1)
    #s4_1 = inc_block_a(s4_1)

    #s4_1 = reduction_block_a(s4_1)
    s4_1 = conv_block(s4_1, 512)
    s4=Concatenate()([s4,s4_1])



    b4 = conv_block(p4, 1024)  # Bridge

    d1 = decoder_block_con(b4, s4, 512)
    d2 = decoder_block_con(d1, s3, 256)
    d3 = decoder_block_con(d2, s2, 128)
    d4 = decoder_block_con(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  # Binary (can be multiclass)

    model = Model(inputs, outputs, name="Aim_Unet")
    return model
