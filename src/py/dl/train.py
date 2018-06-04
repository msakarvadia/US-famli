
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import u_nn as nn

import os
from datetime import datetime
import json
import glob

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='U network for segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

in_group = parser.add_mutually_exclusive_group(required=True)
  
in_group.add_argument('--tfrecords', type=str, help='tfrecords filename')
in_group.add_argument('--json', type=str, help='json file with obj {"shape": [], "tfrecords": "pathtofile"}')

parser.add_argument('--shape', nargs='+', type=int, help='Shape for images in tfrecords')
parser.add_argument('--out', help='Output dirname', default="./out")
parser.add_argument('--model', help='Output modelname, the output name will be <outdir>/model-<num step>', default="model")
parser.add_argument('--keep_prob', help='The probability that each element is kept during training', type=float, default=1.0)
parser.add_argument('--learning_rate', help='Learning rate, default=1e-5', type=float, default=1e-5)
parser.add_argument('--decay_rate', help='decay rate, default=0.96', type=float, default=0.96)
parser.add_argument('--decay_steps', help='decay steps, default=10000', type=int, default=1000)
parser.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=8)
parser.add_argument('--num_epochs', help='Number of epochs', type=int, default=10)
parser.add_argument('--num_labels', help='Number of labels for the softmax output', type=int, default=2)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()

tfrecords_arr = []

if(args.json):
  with open(args.json, "r") as f:
    obj = json.load(f)

    tfrecords_dir = os.path.join(os.path.dirname(args.json), obj["tfrecords"], '**/*.tfrecord')
    for tfr in glob.iglob(tfrecords_dir, recursive=True):
      tfrecords_arr.append(tfr)

    image_shape = obj['image_shape']
    label_shape = obj['label_shape']

else:
  tfrecords_arr.append(args.tfrecords)

outvariablesdirname = args.out
modelname = args.model
k_prob = args.keep_prob
learning_rate = args.learning_rate
decay_rate = args.decay_rate
decay_steps = args.decay_steps
batch_size = args.batch_size
num_epochs = args.num_epochs
num_labels = args.num_labels
ps_device = args.ps_device
w_device = args.w_device

print('tfrecords', args.tfrecords)
print('json', args.json)
print('keep_prob', k_prob)
print('learning_rate', learning_rate)
print('decay_rate', decay_rate)
print('decay_steps', decay_steps)
print('batch_size', batch_size)
print('num_epochs', num_epochs)
print('num_labels', num_labels)
print('ps_device', ps_device)
print('w_device', w_device)

graph = tf.Graph()

with graph.as_default():

  iterator = nn.inputs(batch_size=batch_size,
    num_epochs=num_epochs,
    filenames=tfrecords_arr,
    num_labels=num_labels,
    image_shape=image_shape, 
    label_shape=label_shape)

  x, y_ = iterator.get_next()
  
  keep_prob = tf.placeholder(tf.float32)

  y_conv = nn.inference(x, num_labels=num_labels, keep_prob=keep_prob, is_training=True, ps_device=ps_device, w_device=w_device)

  # calculate the loss from the results of inference and the labels
  # weights = tf.constant(np.array([1, 99999], dtype=np.float32))
  weights = None
  loss = nn.loss(y_conv, y_, class_weights=weights)

  tf.summary.scalar("loss", loss)

  train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)
  
  
  accuracy_eval,auc_eval,fn_eval,fp_eval,tn_eval,tp_eval = nn.metrics(y_conv, y_, class_weights=weights)
  tf.summary.scalar("accuracy_0", accuracy_eval[0])
  tf.summary.scalar("accuracy_1", accuracy_eval[1])
  tf.summary.scalar("auc_0", auc_eval[0])
  tf.summary.scalar("auc_1", auc_eval[1])
  tf.summary.scalar("fn_0", fn_eval[0])
  tf.summary.scalar("fn_1", fn_eval[1])
  tf.summary.scalar("fp_0", fp_eval[0])
  tf.summary.scalar("fp_1", fp_eval[1])
  tf.summary.scalar("tn_0", tn_eval[0])
  tf.summary.scalar("tn_1", tn_eval[1])
  tf.summary.scalar("tp_0", tp_eval[0])
  tf.summary.scalar("tp_1", tp_eval[1])


  summary_op = tf.summary.merge_all()

  with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    # specify where to write the log files for import to TensorBoard
    now = datetime.now()
    summary_writer = tf.summary.FileWriter(os.path.join(outvariablesdirname, now.strftime("%Y%m%d-%H%M%S")), sess.graph)

    sess.run([iterator.initializer])
    step = 0

    while True:
      try:

        _, loss_value, summary, accuracy, auc, fn, fp, tn, tp = sess.run([train_step, loss, summary_op, accuracy_eval, auc_eval, fn_eval, fp_eval, tn_eval, tp_eval], feed_dict={keep_prob: k_prob})

        if step % 100 == 0:
          print('OUTPUT: Step %d: loss = %.3f' % (step, loss_value))
          print('ACCURACY = %.3f, AUC = %.3f, FN = %.3f, FP = %.3f, TN = %.3f, TP = %.3f' % (accuracy[0], auc[0], fn[0], fp[0], tn[0], tp[0]))
          # output some data to the log files for tensorboard
          summary_writer.add_summary(summary, step)
          summary_writer.flush()

          # less frequently output checkpoint files.  Used for evaluating the model
        if step % 1000 == 0:
          save_path = saver.save(sess, os.path.join(outvariablesdirname, modelname), global_step=step)
          # sess.run([valid_iterator.initializer])
          # while True:
          #   try:
          #     batch_valid_data, batch_valid_labels = sess.run([next_valid_data, next_valid_labels])
          #     _, accuracy, auc = sess.run([y_conv, accuracy_eval, auc_eval], feed_dict={x: batch_valid_data, y_: batch_valid_labels, keep_prob: 1})
          #     print('Validation accuracy = %.3f, Auc = %.3f ' % (accuracy[0], auc[0]))
          #   except tf.errors.OutOfRangeError:
          #     break

        step += 1

      except tf.errors.OutOfRangeError:
        break

    outmodelname = os.path.join(outvariablesdirname, modelname)
    print('Step:', step)
    print('Saving model:', outmodelname)
    saver.save(sess, outmodelname, global_step=step)
