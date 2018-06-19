#include "sample_imageCLP.h"

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

using namespace std;

typedef unsigned short PixelType;
static const int Dimension = 3;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef InputImageType::IndexType InputImageIndexType;
typedef InputImageType::Pointer InputImagePointerType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::ImageRegionIterator<InputImageType> InputRegionIteratorType;
typedef itk::NeighborhoodIterator<InputImageType> InputIteratorType;
typedef InputIteratorType::RadiusType InputIteratorRadiusType;
typedef itk::ImageRandomNonRepeatingConstIteratorWithIndex<InputImageType> InputRandomIteratorType;
typedef itk::ImageFileWriter<InputImageType> ImageFileWriterType;

typedef unsigned short VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, Dimension> VectorImageType;  
typedef itk::ComposeImageFilter< InputImageType, VectorImageType> ComposeImageFilterType;
typedef itk::NeighborhoodIterator<VectorImageType> VectorImageIteratorType;
typedef VectorImageIteratorType::RadiusType VectorImageRadiusType;
typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;

bool containsLabel(InputImagePointerType labelImage, InputIteratorRadiusType radius, InputImageIndexType index, int labelValueContains, double labelValueContainsPercentageMax){
	if(labelImage && labelValueContains > 0){
		int count = 0;

		InputIteratorType init(radius, labelImage, labelImage->GetLargestPossibleRegion());
		init.SetLocation(index);

		int size = init.Size();
		
		for(int i = 0; i < size; i++){
			if(init.GetPixel(i) == labelValueContains){
				count++;	
			}
		}
		double ratio = ((double)count)/((double)size);
		return  ratio <= labelValueContainsPercentageMax;	
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

	VectorImageType::Pointer vectorcomposeimage = 0;

	if(composeImages){
		ComposeImageFilterType::Pointer composeImageFilter = ComposeImageFilterType::New();
		for(int i = 0; i < inputImagesVector.size(); i++){
			composeImageFilter->SetInput(i, inputImagesVector[i]);
		}
		composeImageFilter->Update();

		vectorcomposeimage = composeImageFilter->GetOutput();
	}

	InputImageType::Pointer labelImage = 0;

	if(labelImageFilename.compare("") != 0){
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
		readimage->SetFileName(labelImageFilename);
		readimage->Update();
		labelImage = readimage->GetOutput();	
	}

	InputImageType::Pointer maskImage = 0;

	if(maskImageFilename.compare("") != 0){
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
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
	radius[2] = neighborhood[2];

	VectorImageType::Pointer vectoroutputimage = 0;
	InputImagePointerType outputimage = 0;

	if(composeImages){
		vectoroutputimage = VectorImageType::New();

		VectorImageType::SizeType size;
		size[0] = radius[0]*2 + 1;
		size[1] = radius[1]*2 + 1;
		size[2] = radius[2]*2 + 1;
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
		size[2] = radius[2]*2 + 1;
		VectorImageType::RegionType region;
		region.SetSize(size);
		
		outputimage->SetRegions(region);
		outputimage->SetSpacing(inputImagesVector[0]->GetSpacing());
		outputimage->SetDirection(inputImagesVector[0]->GetDirection());
		outputimage->Allocate();
	}

	char *uuid = new char[100];
	
	while(!randomit.IsAtEnd() && numberOfSamples){
		if(containsLabel(labelImage, radius, randomit.GetIndex(), labelValueContains, labelValueContainsPercentageMax) && (!maskImage || (maskImage && maskImage->GetPixel(randomit.GetIndex())))){
			uuid_t id;
			uuid_generate(id);
		  	uuid_unparse(id, uuid);

		  	numberOfSamples--;

		  	InputImageType::PointType outorigin;

			VectorImageType::IndexType outputoriginindex = randomit.GetIndex();
			outputoriginindex[0] -= neighborhood[0];
			outputoriginindex[1] -= neighborhood[1];
			outputoriginindex[2] -= neighborhood[2];

		  	if(vectoroutputimage){

		  		vectorcomposeimage->TransformIndexToPhysicalPoint(outputoriginindex, outorigin);
		  		vectoroutputimage->SetOrigin(outorigin);

	  			VectorImageIteratorType imgit(radius, vectorcomposeimage, vectorcomposeimage->GetLargestPossibleRegion());
	  			imgit.SetLocation(randomit.GetIndex());

	  			VectorImageIteratorType outit(radius, vectoroutputimage, vectoroutputimage->GetLargestPossibleRegion());
	  			
	  			VectorImageType::IndexType outputindex;
				outputindex[0] = neighborhood[0];
				outputindex[1] = neighborhood[1];
				outputindex[2] = neighborhood[2];

				outit.SetLocation(outputindex);

				string outfilename = outputImageDirectory;

			  	outfilename.append(string(uuid)).append("-img").append(".nrrd");
				
				for(int j = 0; j < imgit.Size(); j++){
					outit.SetPixel(j, imgit.GetPixel(j));
				}

				cout<<"Writing file: "<<outfilename<<endl;
				VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
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
					outputindex[2] = neighborhood[2];

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