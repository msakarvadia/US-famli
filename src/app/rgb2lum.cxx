
#include "rgb2lumCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>

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
    static const int dimension = 2;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;

    typedef itk::Image< itk::RGBPixel<InputPixelType>, dimension> InputRGBImageType;
    typedef itk::ImageFileReader< InputRGBImageType > InputImageRGBFileReaderType;

    typedef itk::Image< double, dimension> OutputComponentImageType;

    typedef unsigned short OutputPixelType;
    typedef itk::Image< OutputPixelType, dimension> OutputImageType;

    typedef itk::ImageFileWriter< OutputImageType > OutputImageFileWriterType;
    
    InputImageRGBFileReaderType::Pointer reader = InputImageRGBFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    InputImageType::Pointer outimg;

    if(extractComponent == -1){
        typedef itk::RGBToLuminanceImageFilter< InputRGBImageType, InputImageType > RGBToLuminanceImageFilterType;

        RGBToLuminanceImageFilterType::Pointer lumfilter = RGBToLuminanceImageFilterType::New();
        lumfilter->SetInput(reader->GetOutput());
        lumfilter->Update();
        outimg = lumfilter->GetOutput();
        
    }else{

        typedef itk::VectorIndexSelectionCastImageFilter<InputRGBImageType, InputImageType> VectorIndexSelectionCastImageFilterType;
        VectorIndexSelectionCastImageFilterType::Pointer vector_selection = VectorIndexSelectionCastImageFilterType::New();
        vector_selection->SetIndex(extractComponent);
        vector_selection->SetInput(reader->GetOutput());
        vector_selection->Update();
        outimg = vector_selection->GetOutput();
    }

    typedef itk::RescaleIntensityImageFilter<InputImageType, InputImageType> RescaleIntensityImageFilterType;
    RescaleIntensityImageFilterType::Pointer rescale = RescaleIntensityImageFilterType::New();
    rescale->SetInput(outimg);
    rescale->SetOutputMinimum(outputMinimum);
    rescale->SetOutputMaximum(outputMaximum);
    rescale->Update();

    typedef itk::CastImageFilter< InputImageType, OutputImageType > CastImageFilterType;

    CastImageFilterType::Pointer cast = CastImageFilterType::New();

    cast->SetInput(rescale->GetOutput());
    cast->Update();

    OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->UseCompressionOn();
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(cast->GetOutput());
    writer->Update();   


    return EXIT_SUCCESS;
}
