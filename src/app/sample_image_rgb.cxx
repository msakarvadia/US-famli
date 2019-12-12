#include "sample_image_rgbCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkComposeImageFilter.h>
#include <itkVectorImage.h>
#include <itkImageFileWriter.h>
#include <itkNeighborhoodIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageRandomNonRepeatingIteratorWithIndex.h>
#include <itkFixedArray.h>

#include <uuid/uuid.h>
#include <iostream>
#include <string>

#include <itkConnectedComponentImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkLabelImageToLabelMapFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkRGBPixel.h>

using namespace std;

typedef itk::RGBPixel<unsigned char> PixelType;
static const int Dimension = 2;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef InputImageType::IndexType InputImageIndexType;
typedef InputImageType::Pointer InputImagePointerType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::ImageRegionIterator<InputImageType> InputRegionIteratorType;
typedef itk::NeighborhoodIterator<InputImageType> InputIteratorType;
typedef InputIteratorType::RadiusType InputIteratorRadiusType;
typedef itk::ImageRandomNonRepeatingConstIteratorWithIndex<InputImageType> InputRandomIteratorType;
typedef itk::ImageFileWriter<InputImageType> ImageFileWriterType;

typedef itk::Image<unsigned char, Dimension> InputImageLabelType;
typedef itk::ImageFileReader<InputImageLabelType> InputImageLabelFileReaderType;
typedef InputImageLabelType::IndexType InputImageLabelIndexType;
typedef InputImageLabelType::Pointer InputImageLabelPointerType;
typedef itk::ImageFileReader<InputImageLabelType> InputImageLabelFileReaderType;
typedef itk::NeighborhoodIterator<InputImageLabelType> InputLabelIteratorType;
typedef InputLabelIteratorType::RadiusType InputIteratorLabelRadiusType;

typedef itk::VectorImage<PixelType, Dimension> VectorImageType;  
typedef itk::ComposeImageFilter< InputImageType, VectorImageType> ComposeImageFilterType;
typedef itk::NeighborhoodIterator<VectorImageType> VectorImageIteratorType;
typedef VectorImageIteratorType::RadiusType VectorImageRadiusType;
typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;

bool containsLabel(InputImageLabelPointerType labelImage, InputIteratorLabelRadiusType radius, InputImageLabelIndexType index, int labelValueContains, double labelValueContainsPercentageMax, double labelValueContainsPercentageMin = 0){
	if(labelImage && labelValueContains >= 0){
		int count = 0;

		InputLabelIteratorType init(radius, labelImage, labelImage->GetLargestPossibleRegion());
		init.SetLocation(index);

		int size = init.Size();
		
		for(int i = 0; i < size; i++){
			if(init.GetPixel(i) == labelValueContains){
				count++;	
			}
		}
		double ratio = ((double)count)/((double)size);
		return  labelValueContainsPercentageMin <= ratio && ratio <= labelValueContainsPercentageMax;	
	}

	return true;
}

int main (int argc, char * argv[]){


	PARSE_ARGS;
	
	if(vectorImageFilename.size() == 0 || (labelImageFilename.compare("") == 0)){
		cout<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
		return 1;
	}

	if(outputImageDirectory.compare("") != 0){
		outputImageDirectory.append("/");
	}

	vector< InputImagePointerType > inputImagesVector; 

	for(int i = 0; i < vectorImageFilename.size(); i++){
		cout<<"Reading:"<<vectorImageFilename[i]<<endl;
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
		readimage->SetFileName(vectorImageFilename[i]);
		readimage->Update();
		inputImagesVector.push_back(readimage->GetOutput());
	}
	
	if(labelValueContains != -1){
		cout<<"The region contains label: "<<labelValueContains<<", ratiomax: "<<labelValueContainsPercentageMax<<endl;
	}

	VectorImageType::Pointer vectorcomposeimage;

	if(composeImages){
		ComposeImageFilterType::Pointer composeImageFilter = ComposeImageFilterType::New();
		for(int i = 0; i < inputImagesVector.size(); i++){
			composeImageFilter->SetInput(i, inputImagesVector[i]);
		}
		composeImageFilter->Update();

		vectorcomposeimage = composeImageFilter->GetOutput();
	}

	InputImageLabelType::Pointer labelImage;

	if(labelImageFilename.compare("") != 0){
		InputImageLabelFileReaderType::Pointer readimage = InputImageLabelFileReaderType::New();
		readimage->SetFileName(labelImageFilename);
		readimage->Update();
		labelImage = readimage->GetOutput();	
	}

	InputImageLabelType::Pointer maskImage;

	if(maskImageFilename.compare("") != 0){
		InputImageLabelFileReaderType::Pointer readimage = InputImageLabelFileReaderType::New();
		readimage->SetFileName(maskImageFilename);
		readimage->Update();
		maskImage = readimage->GetOutput();	
	}

	if(inputImagesVector.size() == 0){
		cerr<<"You should totally not see this, stop breaking the code"<<endl;
	}
		
	InputRandomIteratorType randomit(inputImagesVector[0], inputImagesVector[0]->GetLargestPossibleRegion());
	randomit.SetNumberOfSamples(inputImagesVector[0]->GetLargestPossibleRegion().GetNumberOfPixels());
	randomit.GoToBegin();

	VectorImageIteratorType::RadiusType radius;	
	radius[0] = neighborhood[0];
	radius[1] = neighborhood[1];
	// radius[2] = neighborhood[2];

	VectorImageType::Pointer vectoroutputimage;
	InputImagePointerType outputimage;

	if(composeImages){
		vectoroutputimage = VectorImageType::New();

		VectorImageType::SizeType size;
		size[0] = radius[0]*2 + 1;
		size[1] = radius[1]*2 + 1;
		// size[2] = radius[2]*2 + 1;
		VectorImageType::RegionType region;
		region.SetSize(size);
		
		vectoroutputimage->SetRegions(region);
		vectoroutputimage->SetVectorLength(vectorImageFilename.size());
		vectoroutputimage->SetSpacing(vectorcomposeimage->GetSpacing());
		vectoroutputimage->SetDirection(vectorcomposeimage->GetDirection());
		vectoroutputimage->Allocate();
	}else{
		outputimage = InputImageType::New();

		InputImageType::SizeType size;
		size[0] = radius[0]*2 + 1;
		size[1] = radius[1]*2 + 1;
		// size[2] = radius[2]*2 + 1;
		VectorImageType::RegionType region;
		region.SetSize(size);
		
		outputimage->SetRegions(region);
		outputimage->SetSpacing(inputImagesVector[0]->GetSpacing());
		outputimage->SetDirection(inputImagesVector[0]->GetDirection());
		outputimage->Allocate();
	}

	char *uuid = new char[100];
	
	
	while(!randomit.IsAtEnd() && numberOfSamples){
		if(containsLabel(labelImage, radius, randomit.GetIndex(), labelValueContains, labelValueContainsPercentageMax) && (!maskImage || (maskImage && containsLabel(maskImage, radius, randomit.GetIndex(), maskValue, 1, 1)))){
			uuid_t id;
			uuid_generate(id);
		  	uuid_unparse(id, uuid);

		  	numberOfSamples--;

		  	InputImageType::PointType outorigin;

			VectorImageType::IndexType outputoriginindex = randomit.GetIndex();
			outputoriginindex[0] -= neighborhood[0];
			outputoriginindex[1] -= neighborhood[1];
			// outputoriginindex[2] -= neighborhood[2];

		  	if(vectoroutputimage){

		  		vectorcomposeimage->TransformIndexToPhysicalPoint(outputoriginindex, outorigin);
		  		vectoroutputimage->SetOrigin(outorigin);

	  			VectorImageIteratorType imgit(radius, vectorcomposeimage, vectorcomposeimage->GetLargestPossibleRegion());
	  			imgit.SetLocation(randomit.GetIndex());

	  			VectorImageIteratorType outit(radius, vectoroutputimage, vectoroutputimage->GetLargestPossibleRegion());
	  			
	  			VectorImageType::IndexType outputindex;
				outputindex[0] = neighborhood[0];
				outputindex[1] = neighborhood[1];
				// outputindex[2] = neighborhood[2];

				outit.SetLocation(outputindex);

				string outfilename = outputImageDirectory;

			  	outfilename.append(string(uuid)).append("-img").append(".nrrd");
				
				for(int j = 0; j < imgit.Size(); j++){
					outit.SetPixel(j, imgit.GetPixel(j));
				}

				cout<<"Writing file: "<<outfilename<<endl;
				VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
				writer->UseCompressionOn();
				writer->SetFileName(outfilename);
				writer->SetInput(vectoroutputimage);
				writer->Update();
		  	}
		  	if(outputimage){

		  		for(int i = 0; i < inputImagesVector.size(); i++){

		  			inputImagesVector[i]->TransformIndexToPhysicalPoint(outputoriginindex, outorigin);
					outputimage->SetOrigin(outorigin);

		  			InputIteratorType imgit(radius, inputImagesVector[i], inputImagesVector[i]->GetLargestPossibleRegion());
		  			imgit.SetLocation(randomit.GetIndex());

		  			InputIteratorType outit(radius, outputimage, outputimage->GetLargestPossibleRegion());
		  			
		  			InputImageType::IndexType outputindex;
					outputindex[0] = neighborhood[0];
					outputindex[1] = neighborhood[1];
					// outputindex[2] = neighborhood[2];

					outit.SetLocation(outputindex);

					string outfilename = outputImageDirectory;

					char buffer_i [50];
					sprintf(buffer_i, "%d", i);

				  	outfilename.append(string(uuid)).append("-img").append(buffer_i).append(".nrrd");
					
					for(int j = 0; j < imgit.Size(); j++){
						outit.SetPixel(j, imgit.GetPixel(j));
					}

					cout<<"Writing file: "<<outfilename<<endl;
					ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
					writer->UseCompressionOn();
					writer->SetFileName(outfilename);
					writer->SetInput(outputimage);
					writer->Update();
					
				}
		  	}
		}
		++randomit;
	}

	delete[] uuid;

	

	return 0;
}