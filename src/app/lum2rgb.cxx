
#include "lum2rgbCLP.h"

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
    typedef unsigned short InputPixelType;
    static const int dimension = 2;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;
    typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
    
    InputImageFileReaderType::Pointer reader = InputImageFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    typedef itk::Image< itk::RGBPixel<unsigned char>, dimension> InputRGBImageType;
    typedef itk::CastImageFilter<InputImageType, InputRGBImageType> CastImageFilterType;

    typedef CastImageFilterType::Pointer CastImageFilterPointerType;
    CastImageFilterPointerType castfilter = CastImageFilterType::New();
    castfilter->SetInput(reader->GetOutput());
    castfilter->Update();

    
    typedef itk::ImageFileWriter< InputRGBImageType > InputLabelImageFileWriterType;
    InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->UseCompressionOn();
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(castfilter->GetOutput());
    writer->Update();


    return EXIT_SUCCESS;
}
