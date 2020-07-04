import itk
import numpy as np
import argparse
import os
import glob

def Resample(img_filename, output_size, fit_spacing = False, iso_spacing = False, img_dimension = 2, pixel_dimension = 1):

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

	if args.linear:
		InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
	else:
		InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]

	ResampleType = itk.ResampleImageFilter[VectorImageType, VectorImageType]

	spacing = img.GetSpacing()
	region = img.GetLargestPossibleRegion()
	size = region.GetSize()

	if(fit_spacing):
		output_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
	else:
		output_spacing = spacing

	if(iso_spacing):
		max_spacing = np.max(output_spacing)
		output_spacing = np.ones_like(output_spacing)*max_spacing
	
	resampleImageFilter = ResampleType.New()
	interpolator = InterpolatorType.New()

	resampleImageFilter.SetDefaultPixelValue(zeroPixel)
	resampleImageFilter.SetOutputSpacing(output_spacing)
	resampleImageFilter.SetOutputOrigin(img.GetOrigin())

	resampleImageFilter.SetInterpolator(interpolator)
	resampleImageFilter.SetSize(output_size)
	resampleImageFilter.SetInput(img)
	resampleImageFilter.Update()

	return resampleImageFilter.GetOutput()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Resample an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	in_group = parser.add_mutually_exclusive_group(required=True)

	in_group.add_argument('--img', type=str, help='image to resample')
	in_group.add_argument('--dir', type=str, help='Directory with image to resample')

	parser.add_argument('--size', nargs="+", type=int, help='Output size', required=True)
	parser.add_argument('--linear', type=bool, help='Use linear interpolation.', default=False)
	parser.add_argument('--fit_spacing', type=bool, help='Fit spacing to output', default=False)
	parser.add_argument('--iso_spacing', type=bool, help='Same spacing for resampled output', default=False)
	parser.add_argument('--dimension', type=int, help='Image dimension', default=2)
	parser.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)
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
					fobj["out"] = os.path.splitext(fobj["out"])[0] + args.out_ext
				if not os.path.exists(os.path.dirname(fobj["out"])):
					os.makedirs(os.path.dirname(fobj["out"]))
				filenames.append(fobj)
	else:
		raise "Set img or dir to resample!"

	for fobj in filenames:

		img = Resample(fobj["img"], args.size, args.fit_spacing, args.iso_spacing, args.dimension, args.pixel_dimension)

		print("Writing:", fobj["out"])
		WriterType = itk.ImageFileWriter[img]
		writer = WriterType.New()
		writer.SetInput(img)
		writer.SetFileName(fobj["out"])
		writer.Update()
