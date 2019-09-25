
#include "rgb2rgbCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;

    const int dimension = 3;
    typedef itk::Image< itk::RGBPixel<unsigned short>, dimension> InputRGBImageType;
    typedef itk::ImageFileReader< InputRGBImageType > InputImageRGBFileReaderType;
    
    InputImageRGBFileReaderType::Pointer reader = InputImageRGBFileReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    InputRGBImageType::Pointer input_image = reader->GetOutput();

    typedef itk::Image<unsigned short, dimension> InputMaskImageType;
    typedef InputMaskImageType::Pointer InputMaskImagePointerType;

    InputMaskImagePointerType input_mask = 0;

    if(inputMaskFilename.compare("") != 0){
        
        typedef itk::ImageFileReader< InputMaskImageType > InputImageMaskFileReaderType;
        
        InputImageMaskFileReaderType::Pointer reader_mask = InputImageMaskFileReaderType::New();
        reader_mask->SetFileName(inputMaskFilename.c_str());
        reader_mask->Update();

        input_mask = reader_mask->GetOutput();

        typedef itk::MaskImageFilter<InputRGBImageType, InputMaskImageType, InputRGBImageType> MaskImageFilterType;
        MaskImageFilterType::Pointer mask_filter = MaskImageFilterType::New();


        mask_filter->SetInput(input_image);
        mask_filter->SetMaskImage(input_mask);
        mask_filter->Update();

        input_image =  mask_filter->GetOutput();
    }

    if(extractComponent == -1){
        typedef itk::ImageFileWriter< InputRGBImageType > InputLabelImageFileWriterType;
        InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();

        cout<<"Writing: "<<outputImageFilename<<endl;
        writer->SetFileName(outputImageFilename.c_str());
        writer->SetInput(input_image);
        writer->Update();    
    }else{
        typedef itk::Image< unsigned short, dimension> OutputComponentImageType;

        typedef itk::VectorIndexSelectionCastImageFilter<InputRGBImageType, OutputComponentImageType> VectorIndexSelectionCastImageFilterType;
        VectorIndexSelectionCastImageFilterType::Pointer vector_selection = VectorIndexSelectionCastImageFilterType::New();
        vector_selection->SetIndex(0);
        vector_selection->SetInput(input_image);
        vector_selection->Update();
        vector_selection->GetOutput();

        typedef itk::ImageFileWriter< OutputComponentImageType > OutputComponentImageFileWriterType;
        OutputComponentImageFileWriterType::Pointer component_writer = OutputComponentImageFileWriterType::New();

        cout<<"Writing: "<<outputImageFilename<<endl;
        component_writer->SetFileName(outputImageFilename.c_str());
        component_writer->SetInput(vector_selection->GetOutput());
        component_writer->Update();    
    }
    

    


    return EXIT_SUCCESS;
}
