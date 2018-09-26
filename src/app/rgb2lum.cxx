
#include "rgb2lumCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkCastImageFilter.h>

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;

    //Read Image
    typedef double InputPixelType;
    static const int dimension = 3;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;
    

    typedef itk::Image< itk::RGBPixel<double>, dimension> InputRGBImageType;
    typedef itk::ImageFileReader< InputRGBImageType > InputImageRGBFileReaderType;
    
    InputImageRGBFileReaderType::Pointer reader = InputImageRGBFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    typedef itk::RGBToLuminanceImageFilter< InputRGBImageType, InputImageType > RGBToLuminanceImageFilterType;

    RGBToLuminanceImageFilterType::Pointer lumfilter = RGBToLuminanceImageFilterType::New();
    lumfilter->SetInput(reader->GetOutput());
    lumfilter->Update();
    InputImagePointerType outputimg = lumfilter->GetOutput();


    typedef unsigned char OutputPixelType;
    typedef itk::Image< OutputPixelType, dimension> OutputImageType;


    typedef itk::CastImageFilter< InputImageType, OutputImageType > CastImageFilterType;

    CastImageFilterType::Pointer cast = CastImageFilterType::New();

    cast->SetInput(outputimg);
    cast->Update();


    typedef itk::ImageFileWriter< OutputImageType > OutputImageFileWriterType;
    OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(cast->GetOutput());
    writer->Update();


    return EXIT_SUCCESS;
}
