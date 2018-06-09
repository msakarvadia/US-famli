
#include "rgb2lumCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRGBToLuminanceImageFilter.h>

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;

    //Read Image
    typedef unsigned char InputPixelType;
    static const int dimension = 3;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;
    

    typedef itk::Image< itk::RGBPixel<unsigned char>, dimension> InputRGBImageType;
    typedef itk::ImageFileReader< InputRGBImageType > InputImageRGBFileReaderType;
    
    InputImageRGBFileReaderType::Pointer reader = InputImageRGBFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    typedef itk::RGBToLuminanceImageFilter< InputRGBImageType, InputImageType > RGBToLuminanceImageFilterType;

    RGBToLuminanceImageFilterType::Pointer lumfilter = RGBToLuminanceImageFilterType::New();
    lumfilter->SetInput(reader->GetOutput());
    lumfilter->Update();
    InputImagePointerType outputimg = lumfilter->GetOutput();


    typedef itk::ImageFileWriter< InputImageType > InputLabelImageFileWriterType;
    InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(outputimg);
    writer->Update();


    return EXIT_SUCCESS;
}
