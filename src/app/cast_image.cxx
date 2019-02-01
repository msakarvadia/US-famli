
#include "cast_imageCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
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


    typedef unsigned char OutputPixelType;
    typedef itk::Image< OutputPixelType, dimension> OutputImageType;

    typedef itk::CastImageFilter< InputImageType, OutputImageType > CastImageFilterType;

    CastImageFilterType::Pointer cast = CastImageFilterType::New();

    cast->SetInput(reader->GetOutput());
    cast->Update();


    typedef itk::ImageFileWriter< OutputImageType > OutputImageFileWriterType;
    OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(cast->GetOutput());
    writer->Update();


    return EXIT_SUCCESS;
}
