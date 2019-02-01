
from __future__ import print_function
import numpy as np
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import itk
import sys
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp

parser = argparse.ArgumentParser(description='Model evaluation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--csv', type=str, help='csv file')

args = parser.parse_args()

def read_image(filename):
  img_read = itk.ImageFileReader.New(FileName=filename)
  img_read.Update()
  img = img_read.GetOutput()
  
  img_np = itk.GetArrayViewFromImage(img).astype(float)

  return img_np

iou_arr = []

if(args.csv):
  with open(args.csv, "r") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    row_keys = csv_reader.fieldnames.copy()

    for row in csv_reader:
      seg_gt = row["seg1"]
      seg_pr = row["seg2"]

      seg_gt_np = read_image(seg_gt).reshape(-1)
      seg_pr_np = read_image(seg_pr).reshape(-1)

      intersection = 2.0 * np.sum(seg_gt_np * seg_pr_np) + 1e-7
      union = np.sum(seg_gt_np) + np.sum(seg_pr_np) + 1e-7
      iou_arr.append(intersection/union)
else:
  with open("iou.json", "r") as f:

    iou_arr = json.load(f)["iou"]

iou_fig_polar = plt.figure()
ax = iou_fig_polar.add_subplot(111, projection='polar')
theta = 2 * np.pi * np.arange(len(iou_arr))/len(iou_arr)
colors = iou_arr
ax.scatter(theta, iou_arr, c=colors, cmap='autumn', alpha=0.75)
ax.set_rlim(0,1)
plt.title('Intersection over union')
locs, labels = plt.xticks()
plt.xticks(locs, np.arange(0, len(iou_arr), round(len(iou_arr)/len(locs))))
iou_fig_polar.savefig("iou_polar.png")

iou_fig = plt.figure()
x_samples = np.arange(len(iou_arr))
plt.scatter(x_samples, iou_arr, c=colors, cmap='autumn', alpha=0.75)
plt.title('Intersection over union')
iou_mean = [np.mean(iou_arr)]*len(iou_arr)
mean_line = plt.plot(x_samples,iou_mean, label='Mean', linestyle='--')
iou_stdev = [np.std(iou_arr)]*len(iou_arr)
stdev_line = plt.plot(x_samples,iou_mean + iou_stdev, label='Mean', linestyle=':', alpha=0.75)
stdev_line = plt.plot(x_samples,iou_mean - iou_stdev, label='Mean', linestyle=':', alpha=0.75)
iou_fig.savefig("iou.png")