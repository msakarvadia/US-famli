
#include "rgb2rgbCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkNumericTraits.h>

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;

    const int dimension = 2;

    typedef double InputPixelType;
    typedef itk::RGBPixel<InputPixelType> InputRGBPixelType;
    typedef itk::Image<InputRGBPixelType , dimension> InputRGBImageType;
    typedef itk::ImageFileReader< InputRGBImageType > InputImageRGBFileReaderType;

    typedef unsigned char OutputPixelType;
    typedef itk::RGBPixel<OutputPixelType> OutputRGBPixelType;
    typedef itk::Image<OutputRGBPixelType , dimension> OutputRGBImageType;
    
    InputImageRGBFileReaderType::Pointer reader = InputImageRGBFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    InputRGBImageType::Pointer input_image = reader->GetOutput();    

    OutputRGBImageType::Pointer output_image = OutputRGBImageType::New();
    output_image->SetRegions(input_image->GetLargestPossibleRegion());
    output_image->SetDirection(input_image->GetDirection());
    output_image->SetSpacing(input_image->GetSpacing());
    output_image->SetOrigin(input_image->GetOrigin());
    output_image->Allocate();

    typedef itk::ImageRegionIterator<InputRGBImageType> InputRGBImageIteratorType;
    typedef itk::ImageRegionIterator<OutputRGBImageType> OutputRGBImageIteratorType;

    InputRGBImageIteratorType it(input_image, input_image->GetLargestPossibleRegion());
    it.GoToBegin();

    InputRGBPixelType min_pix_input;
    min_pix_input.Fill(itk::NumericTraits<InputPixelType>::max());
    double max_pix = itk::NumericTraits<InputPixelType>::max();

    InputRGBPixelType max_pix_input;
    max_pix_input.Fill(itk::NumericTraits<InputPixelType>::min());
    double min_pix = itk::NumericTraits<InputPixelType>::min();

    while(!it.IsAtEnd()){
        for(int i = 0; i < min_pix_input.Size(); i++){
            min_pix_input[i] = min(it.Get()[i], min_pix_input[i]);
            max_pix_input[i] = max(it.Get()[i], max_pix_input[i]);
        }
        ++it;
    }

    InputRGBPixelType max_pix_output;
    max_pix_output.Fill(maxIntensityValue);

    InputRGBPixelType min_pix_output;
    min_pix_output.Fill(minIntensityValue);

    InputRGBPixelType scale = (max_pix_output - min_pix_output);

    for(int i = 0; i < scale.Size(); i++){
        scale[i] = scale[i]/(max_pix_input[i] - min_pix_input[i]);
    }

    OutputRGBImageIteratorType ot(output_image, output_image->GetLargestPossibleRegion());

    it.GoToBegin();
    ot.GoToBegin();

    while(!it.IsAtEnd() && !ot.IsAtEnd()){

        InputRGBPixelType outpix = (it.Get() - min_pix_input);
        for(int i = 0; i < scale.Size(); i++){
            outpix[i] *= scale[i];
        }
        ot.Set((OutputRGBPixelType)(outpix + min_pix_output));

        ++it;
        ++ot;
    }

    typedef itk::ImageFileWriter< OutputRGBImageType > OutputImageFileWriterType;
    OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(output_image);
    writer->Update();    


    return EXIT_SUCCESS;
}
