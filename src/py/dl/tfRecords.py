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
from collections import namedtuple
import nrrd3D_2D
import tfRecords_split
from sklearn.utils import class_weight

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
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def main(args):
	

	print("Input csv", args.csv)
	if(args.enumerate):
		print("Enumerate", args.enumerate)
	print("Output", args.out)

	csv_rows = []

	obj = {}

	with open(args.csv) as csvfile:

		csv_reader = csv.DictReader(csvfile)
		row_keys = csv_reader.fieldnames.copy()

		if("tfRecord" in row_keys):
			row_keys.remove("tfRecord")
		if(not "data_keys" in obj):
			obj["data_keys"] = row_keys

		for row in csv_reader:
			# If we need to slice the images, save 2D slices with the same dimensions
			if args.slice:

				obj["slice"] = True

				slice_imgs = []
				slice_csv_headers = []

				# We check if the path exists, then we read it as an image
				for key in row_keys:
					if os.path.exists(row[key]):
						if not key in slice_csv_headers:
							slice_csv_headers.append(key)
						slice_imgs.append(row[key])

				# We generate an object with the corresponding parameters
				slice_obj = {}
				slice_obj["img"] = slice_imgs
				slice_obj["out_csv_headers"] = slice_csv_headers
				slice_obj["out_csv"] = os.path.join(args.out, "slice.csv")
				slice_obj["out"] = os.path.join(args.out, "slice")

				# We convert the dictionary to a namedtuple, a.k.a, python object, i.e., argparse object
				slice_args = namedtuple("Slice", slice_obj.keys())(*slice_obj.values())
				# Call the main of the script
				nrrd3D_2D.main(slice_args)
				
				# We read the saved data and append the properties of the 3D image to all slices
				with open(slice_obj["out_csv"]) as csvfileslice:
					csv_slice_reader = csv.DictReader(csvfileslice)
					for slice_row in csv_slice_reader:
						for rk in row_keys:
							if(not rk in csv_slice_reader.fieldnames):
								slice_row[rk] = row[rk]
						# We now have a csv_rows list with slices instead of 3D volumes
						csv_rows.append(slice_row)
			else:
				csv_rows.append(row)

	if(not os.path.exists(args.out) or not os.path.isdir(args.out)):
		os.makedirs(args.out)

	if(args.bins_column is not None and args.bins is not None):
		y_train = []

		for fobj in csv_rows:
			y_train.append(float(fobj[args.bins_column]))

		hist, bin_edges = np.histogram(y_train, bins=args.bins, range=args.bins_range)
		y_train = np.digitize(y_train, bin_edges)

		unique_classes = np.sort(np.unique(y_train))
		class_weights = np.array(class_weight.compute_class_weight('balanced', unique_classes, y_train))

		unique_classes_obj = {}
		unique_classes_obj_str = {}
		for uc, cw in zip(unique_classes, class_weights):
			unique_classes_obj[uc] = cw
			unique_classes_obj_str[str(uc)] = cw
		
		for fobj, y in zip(csv_rows, y_train):
			fobj["class_weights"] = unique_classes_obj[y]
		
		obj["class_weights"] = {}
		obj["class_weights"]["shape"] = [1]
		obj["class_weights"]["type"] = "tf.float32"
		obj["class_weights"]["weights"] = unique_classes_obj_str
		obj["data_keys"].append("class_weights")

	elif(args.enumerate):
		obj["enumerate"] = args.enumerate
		obj[args.enumerate] = {}
		obj[args.enumerate]["class"] = {}
		#This is the number of classes
		num_class = 0
		print("Enumerate classes...")
		compute_class_weight = True
		y_train = []
		if args.bins is not None:
			for fobj in csv_rows:
				class_name = fobj[args.enumerate]
				y_train.append(float(class_name))
			
			hist, bin_edges = np.histogram(y_train, bins=args.bins, range=args.bins_range)
			y_train = np.digitize(y_train, bin_edges)
			
			for fobj, class_digit in zip(csv_rows, y_train):
				
				fobj[args.enumerate] = class_digit

				if(not class_digit in obj[args.enumerate]["class"]):
					obj[args.enumerate]["class"][str(class_digit)] = int(class_digit)
			
			obj[args.enumerate]["num_class"] = args.bins

		else:
			for fobj in csv_rows:
				class_name = fobj[args.enumerate]
				if not os.path.exists(class_name):
					y_train.append(class_name)

				if os.path.exists(class_name) and compute_class_weight:
					compute_class_weight = False
					#Count number of labels in image
					ImageType = itk.Image[itk.SS, args.image_dimension]
					img_label_read = itk.ImageFileReader[ImageType].New(FileName=class_name)
					img_label_read.Update()
					img_label = img_label_read.GetOutput()

					label_stats = itk.LabelStatisticsImageFilter[type(img_label),type(img_label)].New()
					label_stats.SetInput(img_label)
					label_stats.SetLabelInput(img_label)
					label_stats.Update()
					
					for class_label in label_stats.GetValidLabelValues():
						if(not class_label in obj[args.enumerate]["class"]):
							obj[args.enumerate]["class"][class_label] = num_class
							num_class += 1

				elif(not class_name in obj[args.enumerate]["class"]):
					obj[args.enumerate]["class"][class_name] = num_class
					num_class += 1
			#Put the total of elements for convenience
			obj[args.enumerate]["num_class"] = num_class

		if compute_class_weight:
			obj[args.enumerate]["class_weights"] = {}

			unique_classes = np.unique(y_train)
			class_weights = np.array(class_weight.compute_class_weight('balanced', unique_classes, y_train))
			for i, class_label in enumerate(unique_classes):
				n = obj[args.enumerate]["class"][str(class_label)]
				obj[args.enumerate]["class_weights"][n] = class_weights[i]

		print(obj[args.enumerate])

	for fobj in csv_rows:
		
		feature = {}
		feature_list = {}
		# This seed is only used when images of different sizes are used
		random_delta_seed = None

		try:

			for key in row_keys:

				if(not key in obj):
					obj[key] = {}

				##If the path exists then it will try to read it as an image
				if(isinstance(fobj[key], str) and os.path.exists(fobj[key])):
					print("Reading:", fobj[key])
					if(args.image_dimension == -1):
						img_read = itk.ImageFileReader.New(FileName=fobj[key])
						img_read.Update()
						img = img_read.GetOutput()
					else:
						if(args.image_dimension == 1):
							if(args.pixel_dimension != -1):
								ImageType = itk.Image[itk.Vector[itk.F, args.pixel_dimension], 2]
							else:
								ImageType = itk.VectorImage[itk.F, 2]
						else:
							if(args.pixel_dimension != -1):
								ImageType = itk.Image[itk.Vector[itk.F, args.pixel_dimension], args.image_dimension]
							else:
								ImageType = itk.VectorImage[itk.F, args.image_dimension]

						img_read = itk.ImageFileReader[ImageType].New(FileName=fobj[key])
						img_read.Update()
						img = img_read.GetOutput()
					
					img_np = itk.GetArrayViewFromImage(img).astype(float)

					# Put the shape of the image in the json object if it does not exists. This is done for global information
					img_shape = list(img_np.shape)
					if(img_shape[0] == 1):
						# If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
						img_shape = img_shape[1:]

					if(args.image_dimension == 1):
						img_shape = [s for s in img_shape if s != 1]

					# This is the number of channels, if the number of components is 1, it is not included in the image shape
					# If it has more than one component, it is included in the shape, that's why we have to add the 1
					if(img.GetNumberOfComponentsPerPixel() == 1):
						img_shape = img_shape + [1]

					img_np = img_np.reshape(img_shape)

					if args.flip_x:
						print("Flip x")
						img_np = np.flip(img_np, axis=0)

					if args.flip_y:
						print("Flip y")
						img_np = np.flip(img_np, axis=1)

					if(args.image_dimension == 1):
						img_np = img_np.reshape([s for s in img_np.shape if s != 1])

					if(args.scale_intensity_columns is not None and key in args.scale_intensity_columns):
						print("Scale intensity", key, args.scale_intensity)
						img_np *= args.scale_intensity

					if args.sequence:
						feature_list[key] = tf.train.FeatureList(feature=[_float_feature(frame.reshape(-1).tolist()) for frame in img_np])
					else:
						feature[key] =  _float_feature(img_np.reshape(-1).tolist())

					if(not "shape" in obj[key]):
						print("Shape", key, img_shape)
						obj[key]["shape"] = img_shape
						if args.sequence:
							obj[key]["shape"] = obj[key]["shape"][1:]
							obj[key]["sequence"] = True
							print("sequence", obj[key]["shape"])


					if(not "max" in obj[key]):
						obj[key]["max"] = float(np.max(img_np))
					else:
						obj[key]["max"] = max(obj[key]["max"], float(np.max(img_np)))

					if(not "min" in obj[key]):
						obj[key]["min"] = float(np.min(img_np))
					else:
						obj[key]["min"] = min(obj[key]["min"], float(np.min(img_np)))

					if(not "type" in obj[key]):
						obj[key]["type"] = "tf.float32"
					
				elif(args.enumerate and key == args.enumerate):
					# If its an enumeration, it will save the class enumeration as int. If is a label map, it never reached this step
					# because it was read as an image. 
					class_name = fobj[key]
					class_number = int(obj[args.enumerate]["class"][str(class_name)])

					feature[key] = _int64_feature(class_number)

					if "class_weights" in obj[args.enumerate]:
						cw = float(obj[args.enumerate]["class_weights"][class_number])
						feature["class_weights"] = _float_feature(cw)
						fobj["class_weights"] = cw
						if not "class_weights" in obj:
							obj["class_weights"] = {}
							obj["class_weights"]["shape"] = [1]
							obj["class_weights"]["type"] = "tf.float32"
							obj["data_keys"].append("class_weights")

					if(not "shape" in obj[key]):
						obj[key]["shape"] = [1]

					if(not "type" in obj[key]):
						obj[key]["type"] = "tf.int64"

				else:
					try:
						# If the previous failed, try converting it to float
						feature[key] = _float_feature(np.array([float(fobj[key])]).tolist())

						if(not "shape" in obj[key]):
							obj[key]["shape"] = [1]

						if(not "type" in obj[key]):
							obj[key]["type"] = "tf.float32"
					except:
						# If it fails the try saving it as a bytes feature
						# encode the string
						feature[key] = _bytes_feature(obj[key].encode())

						if(not "shape" in obj[key]):
							obj[key]["shape"] = [1]

						if(not "type" in obj[key]):
							obj[key]["type"] = "tf.string"
		
			if("tfRecord" in fobj):
				record_path = fobj["tfRecord"]
			else:
				record_path = os.path.join(args.out, str(uuid.uuid4()) + ".tfrecord")

			fobj["tfRecord"] = record_path

			writer = tf.io.TFRecordWriter(record_path)

			if args.sequence:
				example = tf.train.SequenceExample(context=tf.train.Features(feature=feature), feature_lists=tf.train.FeatureLists(feature_list=feature_list))
			else:
				example = tf.train.Example(features=tf.train.Features(feature=feature))

			print("Writing record", fobj)

			writer.write(example.SerializeToString())
			writer.close()

		except Exception as e:
			print('\033[91m', "Error converting to tfRecord", e, '\033[0m', file=sys.stderr)
			print(obj, file=sys.stderr)
			print(fobj, file=sys.stderr)
			if args.exit_on_error:
				quit()

	obj['tfrecords'] = os.path.basename(args.out.rstrip(os.sep))
	
	outjson = args.out.rstrip(os.sep) + ".json"
	print("Writing:", outjson)

	print(obj)
	with open(outjson, "w") as f:
		f.write(json.dumps(obj, sort_keys=True, indent=4))
	
	outcsv = None
	if(len(csv_rows) > 0):
		outcsv = os.path.splitext(args.csv)[0] + "_tfRecords.csv"
		print("Writing:", outcsv)
		csv_headers = list(csv_rows[0].keys())
		print(csv_headers)
		with open(outcsv, "w") as f:
			writer = csv.DictWriter(f, fieldnames=csv_headers)
			writer.writeheader()
			for row in csv_rows:
				writer.writerow(row)

	if(args.split > 0 and args.split < 1):
		# We generate an object with the corresponding parameters
		split_obj = {}
		split_obj["json"] = outjson
		split_obj["split"] = args.split
		split_obj["csv"] = outcsv

		# We convert the dictionary to a namedtuple, a.k.a, python object, i.e., argparse object
		split_args = namedtuple("Split", split_obj.keys())(*split_obj.values())
		# Call the main of the script
		tfRecords_split.main(split_args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Writes data to tfrecords format using a CSV file description as input and creates a JSON file with a description. The column headers are used as keys to store in the tfRecord. a row may contain image filenames, categories or other information to save in tfRecords format. If an image column is found, the maximum pixel value for each image is calulated and written into the json file. Additionally, it will save a new csv file indicating which tfRecord corresponds to the row.')
	
	parser.add_argument('--csv', type=str, help='CSV file with dataset information,', required=True)
	parser.add_argument('--scale_intensity', type=float, default=1.0, help="Scale image intensity by this factor before storing as tfRecord")
	parser.add_argument('--scale_intensity_columns', type=str, nargs="+", default=None, help="Column names of images that should be rescaled before storing")
	parser.add_argument('--enumerate', type=str, default=None, help='Column name in CSV. If you are storing a label or category to perform a classification task. If it is an image, it will read the FIRST image in your csv and extract the existing labels.')
	parser.add_argument('--bins_column', type=str, default=None, help='Column name in CSV. If you want to generate bins to compute weights')
	parser.add_argument('--bins', type=int, default=None, help='Number of bins. Use a histogram with a defined number of bins to create the different classes')
	parser.add_argument('--bins_range', type=float, nargs="+", default=None, help='Bins range')
	parser.add_argument('--slice', type=bool, default=False, help="If it is a 3D image, saves slices in all the major axis and the stores them as tfRecords")
	parser.add_argument('--out', type=str, default="./out", help="Output directory")
	parser.add_argument('--split', type=float, default=0, help="Split the data for evaluation. [0-1], 0=no split")
	parser.add_argument('--image_dimension', type=int, default=-1, help="Set image dimension, by default it will try to guess it but you can set this here. Or use it when a type is not wrapped. Ex. RGB double, it will read the image as a float vector image")
	parser.add_argument('--pixel_dimension', type=int, default=-1, help="Set pixel dimension, by default it will try to guess it but you can set this here.")
	parser.add_argument('--flip_x', type=bool, default=False, help="Flip image in the x axis")
	parser.add_argument('--flip_y', type=bool, default=False, help="Flip image in the y axis")
	parser.add_argument('--sequence', type=bool, default=0, help="If the images are sequences and have variable length set this flag")
	parser.add_argument('--exit_on_error', type=int, default=0, help="Exit if error is found when writing a tfRecord")

	args = parser.parse_args()

	main(args)
