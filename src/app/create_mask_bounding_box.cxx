
#include "create_mask_bounding_boxCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkLabelStatisticsImageFilter.h>


using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;

    const int dimension = 2;
    typedef itk::Image< unsigned short, dimension> InputImageType;
    typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
    typedef itk::ImageFileWriter< InputImageType > InputImageFileWriterType;

    typedef itk::Image< unsigned short, dimension> OutputImageType;
    
    InputImageFileReaderType::Pointer reader = InputImageFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    InputImageType::Pointer input_image = reader->GetOutput();
    
    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;

    
    typedef itk::LabelStatisticsImageFilter< InputImageType, InputImageType > LabelStatisticsImageFilterType;
    typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;
    typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
    LabelStatisticsImageFilterType::Pointer labelStatisticsImageFilter = LabelStatisticsImageFilterType::New();
    labelStatisticsImageFilter->SetLabelInput( input_image );
    labelStatisticsImageFilter->SetInput( input_image );
    labelStatisticsImageFilter->Update();

    for(ValidLabelValuesType::const_iterator liit=labelStatisticsImageFilter->GetValidLabelValues().begin() + 1; liit != labelStatisticsImageFilter->GetValidLabelValues().end(); ++liit){
        InputImageType::RegionType region =  labelStatisticsImageFilter->GetRegion( *liit );

        ImageRegionIteratorType mit(input_image, region);
        mit.GoToBegin();
        while(!mit.IsAtEnd()){
            mit.Set(maskingValue);
            ++mit;
        }
    }
    

    cout<<"Writing: "<<outputImageFilename<<endl;

    InputImageFileWriterType::Pointer writer = InputImageFileWriterType::New();
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(input_image);
    writer->Update(); 
    


    return EXIT_SUCCESS;
}
