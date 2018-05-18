import itk
import argparse
import os

def main(args):
	
	InputType = itk.Image[itk.RGBPixel[itk.UC], 3]
	print("Reading:", args.img)
	reader = itk.ImageFileReader[InputType].New(FileName=args.img)
	img = reader.GetOutput()

	lumfilter = itk.RGBToLuminanceImageFilter.New(Input=img)
	lumimg = lumfilter.GetOutput()

	print("Writing:", args.out)
	writer = itk.ImageFileWriter.New(FileName=args.out, Input=lumimg)
	writer.Update()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='Input jpg image', required=True)
	parser.add_argument('--out', type=str, help='Output filename', default="out.nrrd")

	args = parser.parse_args()

	main(args)