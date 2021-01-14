import itk
import numpy as np
import argparse
import os
import glob
import sys
import csv

def MaskTimeSeries(img_filename, img_mask, args):
	
	img_dimension = args.image_dimension
	pixel_dimension = args.pixel_dimension

	if(pixel_dimension == 1):
		zeroPixel = 0
		VectorImageType = itk.Image[itk.F, img_dimension]
	else:
		zeroPixel = np.zeros(pixel_dimension)

		PixelType = itk.Vector[itk.F, pixel_dimension]
		VectorImageType = itk.Image[PixelType, img_dimension]

	print("Reading:", img_filename)
	img_read = itk.ImageFileReader[VectorImageType].New(FileName=img_filename)
	img_read.Update()
	img = img_read.GetOutput()

	img_np = itk.GetArrayViewFromImage(img)

	img_mask_np = itk.GetArrayViewFromImage(img_mask).astype(bool)

	if(args.time_series):
		for sl in img_np[:]:
			sl *= img_mask_np
	else:
		img_np *= img_mask_np

	return img


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Mask with a 2D mask. Works on 3D images with a 2D mask', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	in_group = parser.add_mutually_exclusive_group(required=True)

	in_group.add_argument('--img', type=str, help='image to resample')
	in_group.add_argument('--dir', type=str, help='Directory with image to resample')
	in_group.add_argument('--csv', type=str, help='CSV file with column img with paths to images to resample')

	parser.add_argument('--mask', type=str, default=None, required=True, help='Mask image filename')
	parser.add_argument('--time_series', type=bool, help='Image is a time series', default=0)

	parser.add_argument('--image_dimension', type=int, help='Image dimension', default=3)
	parser.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)

	parser.add_argument('--csv_column', type=str, default='img', help='CSV column name (Only used if flag csv is used)')
	parser.add_argument('--csv_root_path', type=str, default='', help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')
	
	parser.add_argument('--ow', type=int, help='Overwrite', default=1)
	parser.add_argument('--out', type=str, help='Output image/directory', default="./out.nrrd")
	parser.add_argument('--out_ext', type=str, help='Output extension type', default=None)

	args = parser.parse_args()

	filenames = []
	if(args.img):
		fobj = {}
		fobj["img"] = args.img
		fobj["out"] = args.out
		filenames.append(fobj)
	elif(args.dir):
		out_dir = args.out
		normpath = os.path.normpath("/".join([args.dir, '**', '*']))
		for img in glob.iglob(normpath, recursive=True):
			if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
				fobj = {}
				fobj["img"] = img
				fobj["out"] = os.path.normpath(out_dir + "/" + img.replace(args.dir, ''))
				if args.out_ext is not None:
					out_ext = args.out_ext
					if out_ext[0] != ".":
						out_ext = "." + out_ext
					fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext
				if not os.path.exists(os.path.dirname(fobj["out"])):
					os.makedirs(os.path.dirname(fobj["out"]))
				if not os.path.exists(fobj["out"]) or args.ow:
					filenames.append(fobj)
	elif(args.csv):
		replace_dir_name = args.csv_root_path
		with open(args.csv) as csvfile:
			csv_reader = csv.DictReader(csvfile)
			for row in csv_reader:
				fobj = {}
				fobj["img"] = row[args.csv_column]
				fobj["out"] = row[args.csv_column].replace(args.csv_root_path, args.out)
				if args.out_ext is not None:
					out_ext = args.out_ext
					if out_ext[0] != ".":
						out_ext = "." + out_ext
					fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext
				if not os.path.exists(os.path.dirname(fobj["out"])):
					os.makedirs(os.path.dirname(fobj["out"]))
				if not os.path.exists(fobj["out"]) or args.ow:
					filenames.append(fobj)
	else:
		raise "Set img or dir to mask!"

	ImageType = itk.Image[itk.SS, 2]
	img_mask_read = itk.ImageFileReader[ImageType].New(FileName=args.mask)
	img_mask_read.Update()
	mask_img = img_mask_read.GetOutput()

	for fobj in filenames:

		try:
			
			img = MaskTimeSeries(fobj["img"], mask_img, args)

			print("Writing:", fobj["out"])
			WriterType = itk.ImageFileWriter[img]
			writer = WriterType.New()
			writer.SetInput(img)
			writer.SetFileName(fobj["out"])
			writer.UseCompressionOn()
			writer.Update()
		except Exception as e:
			print(e, file=sys.stderr)
