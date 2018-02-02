from paddle.trainer_config_helpers import *

img = data_layer(name='pixel', size=154587)

__conv_0__ = img_conv_layer(
    input=img,
    filter_size=11,
    num_channels=3,
    num_filters=96,
    stride=4,
    padding=0)
cmrnorm0 = img_cmrnorm_layer(
    input=__conv_0__, size=5, scale=0.0001, power=0.75)
pool0 = img_pool_layer(input=cmrnorm0, pool_size=3, stride=2)

conv1 = img_conv_layer(
    input=pool0,
    filter_size=5,
    num_filters=256,
    stride=1,
    padding=2,
    groups=2)
cmrnorm1 = img_cmrnorm_layer(
    input=conv1, size=5, scale=0.0001, power=0.75)
pool1 = img_pool_layer(input=cmrnorm1, pool_size=3, stride=2)

conv2 = img_conv_layer(
    input=pool1,
    filter_size=3,
    num_filters=384,
    stride=1,
    padding=1,
    groups=1)

conv3 = img_conv_layer(
    input=conv2,
    filter_size=3,
    num_filters=384,
    stride=1,
    padding=1,
    groups=2)

conv4 = img_conv_layer(
    input=conv3,
    filter_size=3,
    num_filters=256,
    stride=1,
    padding=1,
    groups=2)

pool4 = img_pool_layer(input=conv4, pool_size=3, stride=2)

fc0 = fc_layer(
    input=pool4,
    size=4096,
    act=ReluActivation(),
    layer_attr=attrs.ExtraLayerAttribute(drop_rate=0.5))

slope0 = slope_intercept_layer(input=fc0, slope=2.0, intercept=0.0)

fc1 = fc_layer(
    input=slope0,
    size=4096,
    act=ReluActivation(),
    layer_attr=attrs.ExtraLayerAttribute(drop_rate=0.5))

slope1 = slope_intercept_layer(input=fc1, slope=2.0, intercept=0.0)

prob = fc_layer(
    input=slope1, size=1000, act=SoftmaxActivation())

outputs(prob)


