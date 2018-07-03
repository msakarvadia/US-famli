import itk
import argparse
import os
import glob
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import sys
import json
import csv
import uuid

def _int64_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(args):
	
	csv_rows = []

	with open(args.csv) as csvfile:
		for row in csv.DictReader(csvfile):
			csv_rows.append(row)

	if(not os.path.exists(args.out) or not os.path.isdir(args.out)):
		os.makedirs(args.out)

	obj = {}
	obj["image_keys"] = []

	for fobj in csv_rows:

		row_keys = list(fobj.keys())
		if("tfRecord" in row_keys):
			row_keys.remove("tfRecord")
		
		feature = {}

		try:

			for key in row_keys:

				if(os.path.exists(fobj[key])):

					if(not key in obj["image_keys"]):
						obj["image_keys"].append(key)

					img_read = itk.ImageFileReader.New(FileName=fobj[key])
					img_read.Update()
					img = img_read.GetOutput()
					
					img_np = itk.GetArrayViewFromImage(img).astype(float)
					feature[key] =  _float_feature(img_np.reshape(-1).tolist())

					shape_key = str(key) + "_shape"

					if(not shape_key in obj):
						# Put the shape of the image in the json object if it does not exists. This is done for global information
						img_shape = list(img_np.shape)

						if(img_shape[0] == 1):
							# If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
							img_shape = img_shape[1:]

						if(img.GetNumberOfComponentsPerPixel() == 1):
							img_shape = img_shape + [1]

						obj[shape_key] = img_shape

					max_key = str(key) + "_max"
					min_key = str(key) + "_min"

					if(not max_key in obj or not min_key in obj):
						obj[max_key] = np.max(img_np)
						obj[min_key] = np.min(img_np)
					else:
						obj[max_key] = max(obj[max_key], np.max(img_np))
						obj[min_key] = min(obj[min_key], np.min(img_np))

				else:
					try:
						# Try converting the data to int, if it is a float it will fail
						feature[key] = _int64_feature(int(fobj[key]))
					except:
						try:
							# If the previous failed, try converting it to float
							feature[key] = _float_feature(float(fobj[key]))
						except:
							# If it fails the try saving it as a bytes feature
							feature[key] = _bytes_feature(int(fobj[key]))
		
			if("tfRecord" in fobj):
				record_path = fobj["tfRecord"]
			else:
				record_path = os.path.join(args.out, str(uuid.uuid4()) + ".tfrecord")

			fobj["tfRecord"] = record_path

			writer = tf.python_io.TFRecordWriter(record_path)
			example = tf.train.Example(features=tf.train.Features(feature=feature))

			print("Writing record", fobj)

			writer.write(example.SerializeToString())
			writer.close()

		except Exception as e:
			print("Error converting to tfRecord", obj, e, file=sys.stderr)
			print("I'll keep on going...")

	obj['tfrecords'] = os.path.basename(args.out.rstrip(os.sep))
	
	outjson = args.out.rstrip(os.sep) + ".json"
	print("Writing:", outjson)
	with open(outjson, "w") as f:
		f.write(json.dumps(obj))
	
	if(len(csv_rows) > 0):
		outcsv = os.path.splitext(args.csv)[0] + "_tfRecords.csv"
		print("Writing:", outcsv)
		csv_headers = list(csv_rows[0].keys())
		with open(outcsv, "w") as f:
			writer = csv.DictWriter(f, fieldnames=csv_headers)
			writer.writeheader()
			for row in csv_rows:
				writer.writerow(row)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--csv', type=str, help='CSV file with dataset information, a row may contain image filenames to save in tfrecords. The column name is used as key to store in tfRecord format. The maximum pixel value for each image column is written to the json file as well as the shape of the first image found.')
	parser.add_argument('--out', type=str, default="./out", help="Output directory")
	parser.add_argument('--resize', type=int, default=None, nargs='+', help="Resize all images. The number of channels is kept the same.")

	args = parser.parse_args()

	main(args)
