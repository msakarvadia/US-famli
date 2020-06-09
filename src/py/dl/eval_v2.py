
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import importlib
import os
from datetime import datetime, time
import json
import glob
from tf_inputs_v2 import *
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='Evaluate a model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

input_param_group = parser.add_argument_group('Input')
input_param_group.add_argument('--json', type=str, help='json file with the description of the inputs, generate it with tfRecords.py', required=True)
input_param_group.add_argument('--model', type=str, help='Directory of exported model', required=True)

eval_param_group = parser.add_argument_group('Evaluation parameters')
eval_param_group.add_argument("--type", help='Type of evaluation of the neural network', default="class")


args = parser.parse_args()

json_tf_records = args.json
model_path = args.model
eval_type = args.type


tf_inputs = TFInputs(json_filename=json_tf_records)

if(eval_type == "class"):
	class_names = tf_inputs.get_class_names()
	print(class_names)

dataset = tf_inputs.tf_inputs(batch_size=1, buffer_size=100)

model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
model.summary()


y_pred_arr = []
y_true_arr = []

fpr_arr = []
tpr_arr = []
roc_auc_arr = []
iou_arr = []

abs_diff_arr = []
mse_arr = []

for image_batch in dataset:
	y_pred = model.predict(image_batch[0])

	if(eval_type == "class"):
		y_pred = np.argmax(np.array(y_pred), axis=1)
		y_pred_arr.extend(y_pred)
		y_true_arr.extend(np.reshape(image_batch[1], -1).tolist())
	elif(eval_type == "segmentation"):
		fpr, tpr, _ = roc_curve(np.array(image_batch[1]).reshape(-1), np.array(y_pred).reshape(-1), pos_label=1)
		roc_auc = auc(fpr,tpr)

		fpr_arr.append(fpr)
		tpr_arr.append(tpr)
		roc_auc_arr.append(roc_auc)

		y_pred_flat = np.array(y_pred).reshape((len(y_pred), -1))
		labels_flat = np.array(image_batch[1]).reshape((len(y_pred), -1))

		for i in range(len(y_pred)):
			intersection = 2.0 * np.sum(y_pred_flat[i] * labels_flat[i]) + 1e-7
			union = np.sum(y_pred_flat[i]) + np.sum(labels_flat[i]) + 1e-7
			iou_arr.append(intersection/union)

	elif(eval_type == "image"):
		abs_diff_arr.extend(np.average(np.absolute(y_pred - image_batch[1]).reshape([batch_size, -1]), axis=-1))
		mse_arr.extend(np.average(np.square(y_pred - image_batch[1]).reshape([batch_size, -1]), axis=-1))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.3f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

if(eval_type == "class"):
  # Compute confusion matrix

  cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
  np.set_printoptions(precision=3)

  # Plot non-normalized confusion matrix
  fig = plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
  confusion_filename = os.path.splitext(json_tf_records)[0] + "_confusion.png"
  fig.savefig(confusion_filename)
  # Plot normalized confusion matrix
  fig2 = plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

  norm_confusion_filename = os.path.splitext(json_tf_records)[0] + "_norm_confusion.png"
  fig2.savefig(norm_confusion_filename)

elif(eval_type == "segmentation"):

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_arr]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(len(fpr_arr)):
      mean_tpr += interp(all_fpr, fpr_arr[i], tpr_arr[i])

  mean_tpr /= len(fpr_arr)

  roc_auc = auc(all_fpr, mean_tpr)

  roc_fig = plt.figure()
  lw = 1
  plt.plot(all_fpr, mean_tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")

  roc_filename = os.path.splitext(json_tf_records)[0] + "_roc.png"
  roc_fig.savefig(roc_filename)

  iou_obj = {}
  iou_obj["iou"] = iou_arr

  iou_json = os.path.splitext(json_tf_records)[0] + "_iou_arr.json"

  with open(iou_json, "w") as f:
    f.write(json.dumps(iou_obj))

  iou_fig_polar = plt.figure()
  ax = iou_fig_polar.add_subplot(111, projection='polar')
  theta = 2 * np.pi * np.arange(len(iou_arr))/len(iou_arr)
  colors = iou_arr
  ax.scatter(theta, iou_arr, c=colors, cmap='autumn', alpha=0.75)
  ax.set_rlim(0,1)
  plt.title('Intersection over union')
  locs, labels = plt.xticks()
  plt.xticks(locs, np.arange(0, len(iou_arr), round(len(iou_arr)/len(locs))))

  iou_polar_filename = os.path.splitext(json_tf_records)[0] + "_iou_polar.png"
  iou_fig_polar.savefig(iou_polar_filename)

  iou_fig = plt.figure()
  x_samples = np.arange(len(iou_arr))
  plt.scatter(x_samples, iou_arr, c=colors, cmap='autumn', alpha=0.75)
  plt.title('Intersection over union')
  iou_mean = np.mean(iou_arr)
  plt.plot(x_samples,[iou_mean]*len(iou_arr), label='Mean', linestyle='--')
  plt.text(len(iou_arr) + 2,iou_mean, '%.3f'%iou_mean)
  iou_stdev = np.std(iou_arr)
  stdev_line = plt.plot(x_samples,iou_mean + [iou_stdev]*len(iou_arr), label='Stdev', linestyle=':', alpha=0.75)
  stdev_line = plt.plot(x_samples,iou_mean - [iou_stdev]*len(iou_arr), label='Stdev', linestyle=':', alpha=0.75)
  plt.text(len(iou_arr) + 2,iou_mean + iou_stdev, '%.3f'%iou_stdev, alpha=0.75, fontsize='x-small')
  iou_filename = os.path.splitext(json_tf_records)[0] + "_iou.png"
  iou_fig.savefig(iou_filename)

elif(eval_type == "image"):
  abs_diff_arr = np.array(abs_diff_arr)
  abs_diff_fig = plt.figure()
  x_samples = np.arange(len(abs_diff_arr))
  plt.scatter(x_samples, abs_diff_arr, c=abs_diff_arr, cmap='cool', alpha=0.75, label='Mean absolute error')
  plt.xlabel('Samples')
  plt.ylabel('Absolute error')
  plt.title('Mean absolute error')

  abs_filename = os.path.splitext(json_tf_records)[0] + "_abs_diff.png"
  abs_diff_fig.savefig(abs_filename)

  mse_arr = np.array(mse_arr)
  mse_fig = plt.figure()
  plt.scatter(x_samples, mse_arr, c=mse_arr, cmap='cool', alpha=0.75, label='MSE')
  plt.xlabel('Samples')
  plt.ylabel('MSE')
  plt.title('Mean squared error')

  mse_filename = os.path.splitext(json_tf_records)[0] + "_mse.png"
  mse_fig.savefig(mse_filename)