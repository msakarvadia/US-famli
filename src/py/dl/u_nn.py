from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def read_and_decode(record):

    keys_to_features = {
      "image": tf.FixedLenFeature((), tf.string),
      "label": tf.FixedLenFeature((), tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    
    image = tf.decode_raw(parsed["image"], tf.float32)
    label = tf.decode_raw(parsed["label"], tf.int32)

    return image, label


def inputs(filenames, shape=[], batch_size=1, num_epochs=1, num_labels=2):
    # # inputs: filename, batch_size, num_epochs are scalars
    # # output: image and label pairs for use in training or eval

    # # define the input node
    # with tf.name_scope('input'):
    #     # setup a TF filename_queue
    #     filename_queue = tf.train.string_input_producer(
    #         filenames, num_epochs=num_epochs)

    #     # return and image and label
    #     image, label = read_and_decode(filename_queue, shape)

    #     # shuffle the images, not strictly necessary as the data creating
    #     # phase already did it, but there's no harm doing it again.
    #     images, labels = tf.train.shuffle_batch(
    #         [image, label], batch_size=batch_size, num_threads=4,
    #         capacity=50000,
    #         min_after_dequeue=10000)

    #     labels = tf.one_hot(labels, num_labels)
    #     print_tensor_shape(labels, 'Training labels')

    #     return images, labels

    dataset = tf.data.TFRecordDataset(filenames)
    
    dataset = dataset.map(read_and_decode)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    image, label = iterator.get_next()

    image_re = tf.reshape(image, image_shape)
    label_re = tf.one_hot(tf.reshape(label, label_shape), num_labels)

    print_tensor_shape(image_re, 'image')
    print_tensor_shape(label_re, 'label')

    return image_re, label_re

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())
# x=[batch, in_height, in_width, in_channels]
# filter_shape=[filter_height, filter_width, in_channels, out_channels]
def convolution2d(x, filter_shape, name, strides=[1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
        with tf.device(ps_device):
            w_conv_name = 'w_' + name
            # filter_shape=[in_time,in_channels,out_channels]
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv2d( x, w_conv, strides=strides, padding=padding, name='conv_op' )
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

# x=[batch, height, width, in_channels]
# filter_shape=[height, width, output_channels, in_channels]
def up_convolution2d(x, filter_shape, output_shape, name, strides=[1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
        with tf.device(ps_device):
            w_conv_name = 'w_' + name
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv2d_transpose( x, w_conv, output_shape=output_shape, strides=strides, padding=padding, name='up_conv_op' )
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

# x=[batch, in_height, in_width, in_channels]
# kernel=[k_batch, k_height, k_width, k_channels]
def max_pool(x, name, kernel=[1,3,3,1], strides=[1,2,2,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
        with tf.device(w_device):
            pool_op = tf.nn.max_pool( x, kernel, strides=strides, padding=padding, name='max_pool_op' )
            print_tensor_shape( pool_op, 'max_pool_op shape')

            return pool_op

def convolution(x, filter_shape, name, stride=1, activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            # in_time -> stride in time
            # filter_shape=[in_time,in_channels,out_channels]
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv1d( x, w_conv, stride=stride, padding=padding, name='conv1_op' )
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

def matmul(x, out_channels, name, activation=tf.nn.relu, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):

        in_channels = x.get_shape().as_list()[-1]

    with tf.device(ps_device):
        w_matmul_name = 'w_' + name
        w_matmul = tf.get_variable(w_matmul_name, shape=[in_channels,out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))

        print_tensor_shape( w_matmul, 'w_matmul shape')        

        b_matmul_name = 'b_' + name
        b_matmul = tf.get_variable(name=b_matmul_name, shape=[out_channels])        

    with tf.device(w_device):

        matmul_op = tf.nn.bias_add(tf.matmul(x, w_matmul), b_matmul)

        if(activation):
            matmul_op = activation(matmul_op)

        return matmul_op

def inference(images, num_labels=2, keep_prob=1, batch_size=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

#   input: tensor of images
#   output: tensor of computed logits
    print_tensor_shape(images, "images")

    images = tf.layers.batch_normalization(images, training=is_training)

    shape = images.get_shape().as_list()

    conv1 = convolution2d(images, [3, 3, shape[-1], 64], "conv1_op", strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
    conv1 = convolution2d(conv1, [3, 3, 64, 64], "conv1_op", strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
    pool1 = max_pool(conv1, name="pool1_op")
    # convolution2d(images, [3, 3, 64, 128], "conv2_op", strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
    # convolution2d(images, [3, 3, 128, 256], "conv3_op", strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)

    return pool1

def evaluation(logits, labels, metrics_collections=None):
    #return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)), tf.float32))
    # values, indices = tf.nn.top_k(labels, 1);
    # correct = tf.reshape(tf.nn.in_top_k(logits, tf.cast(tf.reshape( indices, [-1 ] ), tf.int32), 1), [-1] )
    # print_tensor_shape( correct, 'correct shape')
    # return tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    return tf.metrics.accuracy(predictions=tf.argmax(logits, axis=1), labels=tf.argmax(labels, axis=1), name='accuracy', metrics_collections=metrics_collections)

def metrics(logits, labels, metrics_collections=None):
  auc_eval = tf.metrics.auc(predictions=logits, labels=labels, name='auc', metrics_collections=metrics_collections)
  fn_eval = tf.metrics.false_negatives(predictions=logits, labels=labels, name='false_negatives', metrics_collections=metrics_collections)
  fp_eval = tf.metrics.false_positives(predictions=logits, labels=labels, name='false_positives', metrics_collections=metrics_collections)
  tn_eval = tf.metrics.true_negatives(predictions=logits, labels=labels, name='true_negatives', metrics_collections=metrics_collections)
  tp_eval = tf.metrics.true_positives(predictions=logits, labels=labels, name='true_positives', metrics_collections=metrics_collections)
  return auc_eval,fn_eval,fp_eval,tn_eval,tp_eval


def training(loss, learning_rate, decay_steps, decay_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

  # create learning_decay
    # lr = tf.train.exponential_decay( learning_rate,
    #                                  global_step,
    #                                  decay_steps,
    #                                  decay_rate, staircase=True )

    # tf.summary.scalar('2learning_rate', lr )

  # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def loss(logits, labels):
    
    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

    return loss
