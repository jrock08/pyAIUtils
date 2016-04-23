import numpy as np
import pytest
import tensorflow as tf

from context import layers
from context import images

def test_full():
    batch = 1
    in_dim = 2
    out_dim = 3

    input_shape = [batch, in_dim]
    output_shape = [batch, out_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.full(x, out_dim, 'full')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.zeros(input_shape))
    y_ = np.float32(np.zeros(output_shape))
    y_hat = sess.run(y, feed_dict={x: x_})
    assert (np.all(y_hat == y_))


def test_conv2d():
    batch = 1
    height = 3
    width = 3
    filter_size = 3
    in_dim = 4
    out_dim = 5

    input_shape = [batch, height, width, in_dim]
    output_shape = [batch, height, width, out_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.conv2d(x, filter_size, out_dim, 'conv2d')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.zeros(input_shape))
    y_ = np.float32(np.zeros(output_shape))
    y_hat = sess.run(y, feed_dict={x: x_})
    assert (np.all(y_hat == y_))


def test_batch_norm_2d():
    batch = 1
    in_dim = 2
    out_dim = 3

    input_shape = [batch, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.batch_norm(x, '2d')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.random.randn(*input_shape))
    y_hat = sess.run(y, feed_dict={x: x_})

    assert y_hat.shape == x_.shape


def test_batch_norm_4d():
    batch = 1
    width = 2
    height = 3
    in_dim = 4
    out_dim = 5

    input_shape = [batch, width, height, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.batch_norm(x, '4d')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.random.randn(*input_shape))
    y_hat = sess.run(y, feed_dict={x: x_})

    assert y_hat.shape == x_.shape


def test_batch_norm_3d():
    batch = 1
    width = 2
    in_dim = 3
    out_dim = 4

    input_shape = [batch, width, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    with pytest.raises(ValueError):
        y = layers.batch_norm(x, '4d')


def test_resize_image_like():
    batch = 1
    width_height = 5
    width_height2 = 15
    channel = 3
    x = tf.placeholder(tf.float32, [batch, width_height, width_height, channel])
    s = tf.placeholder(tf.float32, [batch, width_height2, width_height2, channel])
    x_rshape = images.resize_images_like(x, s)
    assert x_rshape.get_shape() == s.get_shape()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    x_ = np.float32(np.random.rand(batch, width_height, width_height, channel))
    x_out = sess.run(x_rshape, {x:x_})
    assert x_out.shape == s.get_shape()
