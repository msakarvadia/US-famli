
#include "rgb_maskCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkMaskImageFilter.h>
#include <itkFlatStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>
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
    typedef double InputPixelType;
    typedef itk::Image< itk::RGBPixel<InputPixelType>, dimension> InputRGBImageType;
    typedef itk::ImageFileReader< InputRGBImageType > InputImageRGBFileReaderType;

    typedef double OutputPixelType;
    typedef itk::Image< itk::RGBPixel<OutputPixelType>, dimension> OutputRGBImageType;
    
    InputImageRGBFileReaderType::Pointer reader = InputImageRGBFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    InputRGBImageType::Pointer input_image = reader->GetOutput();
    InputRGBImageType::Pointer output_image;

    typedef itk::Image<unsigned short, dimension> InputMaskImageType;
    typedef InputMaskImageType::Pointer InputMaskImagePointerType;
    typedef itk::ImageRegionIterator< InputMaskImageType > MaskImageRegionIteratorType;
    

    InputMaskImagePointerType input_mask;
        
    typedef itk::ImageFileReader< InputMaskImageType > InputImageMaskFileReaderType;
    
    InputImageMaskFileReaderType::Pointer reader_mask = InputImageMaskFileReaderType::New();
    reader_mask->SetFileName(inputMaskFilename.c_str());
    reader_mask->Update();

    input_mask = reader_mask->GetOutput();

    if(radiusStructuringElement > 0){
        typedef itk::FlatStructuringElement<dimension> StructuringElementType;
        StructuringElementType::RadiusType elementRadius;
        elementRadius.Fill(radiusStructuringElement);

        StructuringElementType structuringElement = StructuringElementType::Box(elementRadius);

        typedef itk::BinaryDilateImageFilter<InputMaskImageType, InputMaskImageType, StructuringElementType> BinaryDilateImageFilterType;

        BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
        dilateFilter->SetInput(input_mask);
        dilateFilter->SetKernel(structuringElement);
        dilateFilter->SetDilateValue(dilateMaskValue);
        dilateFilter->Update();
        input_mask = dilateFilter->GetOutput();
    }

    if(useBoundingBox){
        typedef itk::LabelStatisticsImageFilter< InputMaskImageType, InputMaskImageType > LabelStatisticsImageFilterType;
        typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;
        typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
        LabelStatisticsImageFilterType::Pointer labelStatisticsImageFilter = LabelStatisticsImageFilterType::New();
        labelStatisticsImageFilter->SetLabelInput( input_mask );
        labelStatisticsImageFilter->SetInput( input_mask );
        labelStatisticsImageFilter->Update();

        for(ValidLabelValuesType::const_iterator liit=labelStatisticsImageFilter->GetValidLabelValues().begin() + 1; liit != labelStatisticsImageFilter->GetValidLabelValues().end(); ++liit){
            InputMaskImageType::RegionType region =  labelStatisticsImageFilter->GetRegion( *liit );

            MaskImageRegionIteratorType mit(input_mask, region);
            mit.GoToBegin();
            while(!mit.IsAtEnd()){
                mit.Set(maskingValue);
                ++mit;
            }
        }
    }

    typedef itk::MaskImageFilter<InputRGBImageType, InputMaskImageType, InputRGBImageType> MaskImageFilterType;
    MaskImageFilterType::Pointer mask_filter = MaskImageFilterType::New();

    mask_filter->SetInput(input_image);
    mask_filter->SetMaskImage(input_mask);
    mask_filter->SetMaskingValue(maskingValue);
    mask_filter->Update();

    output_image =  mask_filter->GetOutput();
    

    typedef itk::ImageFileWriter< OutputRGBImageType > OutputImageFileWriterType;
    OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->UseCompressionOn();
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(output_image);
    writer->Update(); 
    


    return EXIT_SUCCESS;
}
