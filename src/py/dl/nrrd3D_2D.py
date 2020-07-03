import itk
import argparse
import os
import numpy as np
import uuid
import sys

def image_save(img_obj, prediction):
  PixelDimension = prediction.shape[-1]
  Dimension = 2
  
  if(PixelDimension < 7):
    if(PixelDimension >= 3 and os.path.splitext(img_obj["out"])[1] not in ['.jpg', '.png']):
      ComponentType = itk.ctype('float')
      PixelType = itk.Vector[ComponentType, PixelDimension]
    elif(PixelDimension == 3):
      PixelType = itk.RGBPixel.UC
      prediction = np.absolute(prediction)
      prediction = np.around(prediction).astype(np.uint16)
    else:
      PixelType = itk.ctype('float')

    OutputImageType = itk.Image[PixelType, Dimension]
    out_img = OutputImageType.New()

  else:

    ComponentType = itk.ctype('float')
    OutputImageType = itk.VectorImage[ComponentType, Dimension]

    out_img = OutputImageType.New()
    out_img.SetNumberOfComponentsPerPixel(PixelDimension)
    
  size = itk.Size[Dimension]()
  size.Fill(1)
  prediction_shape = list(prediction.shape[0:-1])
  prediction_shape.reverse()

  for i, s in enumerate(prediction_shape):
    size[i] = s

  index = itk.Index[Dimension]()
  index.Fill(0)

  RegionType = itk.ImageRegion[Dimension]
  region = RegionType()
  region.SetIndex(index)
  region.SetSize(size)
  
  # out_img.SetRegions(img.GetLargestPossibleRegion())
  out_img.SetRegions(region)
  out_img.SetDirection(img.GetDirection())
  out_img.SetOrigin(img.GetOrigin())
  out_img.SetSpacing(img.GetSpacing())
  out_img.Allocate()

  out_img_np = itk.GetArrayViewFromImage(out_img)
  out_img_np.setfield(np.reshape(prediction, out_img_np.shape), out_img_np.dtype)

  print("Writing:", img_obj["out"])
  writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=out_img)
  writer.UseCompressionOn()
  writer.Update()

def main(args):
	img = itk.imread(args.img)
	img_np = itk.GetArrayViewFromImage(img).astype(float)
	num_splits = img_np.shape[args.axis]
	img_np_split = np.split(img_np, num_splits, axis=args.axis)

	out_shape = np.delete(np.shape(img_np), args.axis)

	PixelType = itk.template(img)[1][0]
	Dimension = 2
	PixelDimension = img.GetNumberOfComponentsPerPixel()
	
	Origin = np.delete(np.array(img.GetOrigin())[::-1], args.axis)
	Spacing = np.delete(np.array(img.GetSpacing())[::-1], args.axis)

	index = itk.Index[Dimension]()
	index.Fill(0)

	size = itk.Size[Dimension]()
	size.Fill(1)
	for i, s in enumerate(np.copy(out_shape)[::-1]):
		size[i] = int(s)

	RegionType = itk.ImageRegion[Dimension]
	Region = RegionType()
	Region.SetIndex(index)
	Region.SetSize(size)

	OutputImageType = itk.VectorImage[PixelType, 2]

	if PixelDimension > 1:
		np.append(out_shape, PixelDimension)

	for i, slice_np in enumerate(img_np_split):
		# print(np.reshape(slice_np, out_shape).shape)
		out_img = OutputImageType.New()
		out_img.SetNumberOfComponentsPerPixel(PixelDimension)
		out_img.SetOrigin(Origin)
		out_img.SetSpacing(Spacing)
		out_img.SetRegions(Region)
		out_img.Allocate()

		out_img_np = itk.GetArrayViewFromImage(out_img)
		out_img_np.setfield(np.reshape(slice_np, out_shape), out_img_np.dtype)

		out_name = os.path.join(args.out, args.prefix + str(i) + args.ext)
		print("Writing:", out_name)
		writer = itk.ImageFileWriter.New(FileName=out_name, Input=out_img)
		writer.UseCompressionOn()
		writer.Update()

	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='Input image', required=True)
	parser.add_argument('--out', type=str, help='Output directory', default="./")
	parser.add_argument('--prefix', type=str, help='Output prefix', default="frame_")
	parser.add_argument('--ext', type=str, help='Output img extension', default=".nrrd")
	parser.add_argument('--axis', type=int, help='Split index', default=0)

	args = parser.parse_args()

	main(args)