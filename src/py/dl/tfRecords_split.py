import argparse
import os
import glob
import numpy as np
import sys
import json
import pandas as pd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_sym(tfrecords_arr, tfrecords_dir, data_description):

	if(not os.path.exists(tfrecords_dir)):
		os.makedirs(tfrecords_dir)

	for tfr in tfrecords_arr:
		try:
			dest = os.path.join(tfrecords_dir, os.path.basename(tfr))
			print("Creating soft link: ", dest)
			os.symlink(tfr, dest)
		except Exception as e:
			print("Error making symlink", e, file=sys.stderr)

	tfrecords_json = tfrecords_dir.rstrip(os.sep) + ".json"
	with open(tfrecords_json, "w") as f:
		f.write(json.dumps(data_description))

def main(args):
	
	json_filename = args.json
	data_description = {}

	with open(json_filename, "r") as f:
		data_description = json.load(f)

	if args.folds and args.csv:

		df = pd.read_csv(args.csv)

		if args.group_by:
			group_by = args.group_by
		else:
			group_by = 'tfrecord'

		unique_group_by = df[group_by].unique()
		samples = round(len(unique_group_by)*args.split)

		print(bcolors.OKBLUE, "Num unique elements:", len(unique_group_by), bcolors.ENDC)
		print(bcolors.OKBLUE, "Num samples for each split: ", samples, bcolors.ENDC)

		np.random.shuffle(unique_group_by)

		start_f = 0
		end_f = samples
		for i in range(args.folds):

			unique_group_test = unique_group_by[start_f:end_f]

			df_train = df[~df[group_by].isin(unique_group_test)]
			df_test = df[df[group_by].isin(unique_group_test)]

			train_fn = json_filename.replace('.json', '_fold' + str(i) + '_train.csv')
			train_fn_dir = train_fn.replace('.csv', '')
			fold_description_train  = {}
			fold_description_train.update(data_description)
			fold_description_train["tfrecords"] = os.path.basename(train_fn_dir)

			create_sym(df_train["tfrecord"], train_fn_dir, fold_description_train)
			print(bcolors.OKGREEN, "Writing:", train_fn, bcolors.ENDC)
			df_train.to_csv(train_fn, index=False)

			test_fn = json_filename.replace('.json', '_fold' + str(i) + '_test.csv')
			print(bcolors.OKGREEN, "Writing:", test_fn, bcolors.ENDC)
			df_test.to_csv(test_fn, index=False)

			start_f += samples
			end_f += samples

	else:

		tfrecords_arr = []
		tfrecords_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"], '**/*.tfrecord')
		for tfr in glob.iglob(tfrecords_dir, recursive=True):
			tfrecords_arr.append(tfr)

		tfrecords_arr = np.array(tfrecords_arr)
		tfrecords_arr = tfrecords_arr[np.random.permutation(tfrecords_arr.size)]

		train_samples = int(tfrecords_arr.size - tfrecords_arr.size*args.split)

		tfrecords_train_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"] + "_split_train")
		data_description_train  = {}
		data_description_train.update(data_description)
		data_description_train["tfrecords"] = os.path.basename(tfrecords_train_dir)

		if(not os.path.exists(tfrecords_train_dir)):
			os.makedirs(tfrecords_train_dir)

		for tfr in tfrecords_arr[0:train_samples]:
			try:
				dest = os.path.join(tfrecords_train_dir, os.path.basename(tfr))
				print("Creating soft link: ", dest)
				os.symlink(tfr, dest)
			except Exception as e:
				print("Error making symlink", e, file=sys.stderr)

		tfrecords_train_json = tfrecords_train_dir.rstrip(os.sep) + ".json"
		with open(tfrecords_train_json, "w") as f:
			f.write(json.dumps(data_description_train))

		tfrecords_eval_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"] + "_split_eval")
		data_description_eval  = {}
		data_description_eval.update(data_description)
		data_description_eval["tfrecords"] = os.path.basename(tfrecords_eval_dir)

		if(not os.path.exists(tfrecords_eval_dir)):
			os.makedirs(tfrecords_eval_dir)

		for tfr in tfrecords_arr[train_samples:]:
			try:
				dest = os.path.join(tfrecords_eval_dir, os.path.basename(tfr))
				print("Creating soft link: ", dest)
				os.symlink(tfr, dest)
			except Exception as e:
				print("Error making symlink", e, file=sys.stderr)
		
		tfrecords_eval_json = tfrecords_eval_dir.rstrip(os.sep) + ".json"
		with open(tfrecords_eval_json, "w") as f:
			f.write(json.dumps(data_description_eval))

		# if(args.csv):
		# 	with open(args.csv) as csvfile:

		# 		csv_reader = csv.DictReader(csvfile)
		# 		row_keys = csv_reader.fieldnames.copy()

		# 		if("tfrecord" in row_keys):
		# 			tf_train_array = []
		# 			tf_eval_array = []
		# 			for row in csv_reader:
		# 				tfrecord_filename = os.path.join(tfrecords_train_dir, os.path.basename(row["tfrecord"]))
		# 				if(os.path.isfile(tfrecord_filename)):
		# 					tf_train_array.append(row)

		# 				tfrecord_filename = os.path.join(tfrecords_eval_dir, os.path.basename(row["tfrecord"]))
		# 				if(os.path.isfile(tfrecord_filename)):
		# 					tf_eval_array.append(row)

		# 			if(len(tf_train_array) > 0):
		# 				outcsv = os.path.splitext(args.csv)[0] + "_train.csv"
		# 				with open(outcsv, "w") as f:
		# 					writer = csv.DictWriter(f, fieldnames=row_keys)
		# 					writer.writeheader()
		# 					for row in tf_train_array:
		# 						writer.writerow(row)

		# 			if(len(tf_eval_array) > 0):
		# 				outcsv = os.path.splitext(args.csv)[0] + "_eval.csv"
		# 				with open(outcsv, "w") as f:
		# 					writer = csv.DictWriter(f, fieldnames=row_keys)
		# 					writer.writeheader()
		# 					for row in tf_eval_array:
		# 						writer.writerow(row)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Splits the data by creating soft links and new folders for training and evaluation purposes')
	
	parser.add_argument('--json', type=str, help='JSON file created by tfRecords.py', required=True)
	parser.add_argument('--split', type=float, default=0, help="Split the data for evaluation. [0-1], 0=no split")
	parser.add_argument('--folds', type=int, help='Split data in folds', default=0)
	parser.add_argument('--csv', type=str, default=None, help="The csv file must contain the column tfrecord, created by tfRecords.py")
	parser.add_argument('--group_by', type=str, help='Column name for grouping criteria', default=None)

	args = parser.parse_args()

	main(args)
