
#include "find_markerCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkLabelImageToLabelMapFilter.h>
#include <itkLabelMapMaskImageFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkNeighborhoodIterator.h>
#include <itkRGBToLuminanceImageFilter.h>

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0 || inputLabelFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;
    if(inputLabelFilename.compare("") != 0){
        cout << "The input label image is: " << inputLabelFilename << endl;
    }

    //Read Image
    typedef unsigned char InputPixelType;
    static const int dimension = 3;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;
    InputImagePointerType imgin = 0;

    if(!lumFilter){

        typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
        typedef InputImageFileReaderType::Pointer InputImageFileReaderPointerType;
        
        InputImageFileReaderPointerType reader = InputImageFileReaderType::New();
        reader->SetFileName(inputImageFilename.c_str());
        reader->Update();

        imgin = reader->GetOutput();
    }else{

        typedef itk::Image< itk::RGBPixel<unsigned char>, dimension> InputRGBImageType;
        typedef itk::ImageFileReader< InputRGBImageType > InputImageRGBFileReaderType;
        
        InputImageRGBFileReaderType::Pointer reader = InputImageRGBFileReaderType::New();
        reader->SetFileName(inputImageFilename.c_str());
        reader->Update();

        typedef itk::RGBToLuminanceImageFilter< InputRGBImageType, InputImageType > RGBToLuminanceImageFilterType;

        RGBToLuminanceImageFilterType::Pointer lumfilter = RGBToLuminanceImageFilterType::New();
        lumfilter->SetInput(reader->GetOutput());
        lumfilter->Update();
        imgin = lumfilter->GetOutput();
    }
    

    
    typedef unsigned short InputLabelPixelType;
    typedef itk::Image< InputLabelPixelType, dimension> InputLabelImageType;
    typedef InputLabelImageType::Pointer InputLabelImagePointerType;

    typedef itk::ImageFileReader< InputLabelImageType > InputLabelImageFileReaderType;
    typedef InputLabelImageFileReaderType::Pointer InputImageLabelFileReaderPointerType;
    InputImageLabelFileReaderPointerType readerlm = InputLabelImageFileReaderType::New();
    readerlm->SetFileName(inputLabelFilename);
    readerlm->Update();
    InputLabelImagePointerType labelimage = readerlm->GetOutput();

    InputLabelImagePointerType outputimg = InputLabelImageType::New();
    outputimg->SetRegions(imgin->GetLargestPossibleRegion());
    outputimg->SetSpacing(imgin->GetSpacing());
    outputimg->SetOrigin(imgin->GetOrigin());
    outputimg->SetDirection(imgin->GetDirection());
    outputimg->Allocate();
    outputimg->FillBuffer(0);

    typedef itk::LabelStatisticsImageFilter< InputImageType, InputLabelImageType > LabelStatisticsImageFilterType;
    LabelStatisticsImageFilterType::Pointer labelStatisticsImageFilter = LabelStatisticsImageFilterType::New();
    labelStatisticsImageFilter->SetLabelInput( labelimage );
    labelStatisticsImageFilter->SetInput(imgin);
    labelStatisticsImageFilter->Update();

    InputLabelImageType::RegionType region =  labelStatisticsImageFilter->GetRegion( 2 );

    std::cout << "region: " << region << std::endl;

    InputImageType::SizeType radius;

    radius[0] = (region.GetSize()[0] - 1)/2;
    radius[1] = (region.GetSize()[1] - 1)/2;
    radius[2] = (region.GetSize()[2] - 1)/2;
    

    typedef itk::NeighborhoodIterator<InputImageType> InputImageNeighborhoodIteratorType;
    typedef itk::NeighborhoodIterator<InputLabelImageType> InputLabelImageNeighborhoodIteratorType;
    

    InputImageNeighborhoodIteratorType it(radius, imgin, imgin->GetLargestPossibleRegion());
    InputLabelImageNeighborhoodIteratorType lbout(radius, outputimg, outputimg->GetLargestPossibleRegion());

    InputLabelImageNeighborhoodIteratorType lbit(radius, labelimage, labelimage->GetLargestPossibleRegion());
    

    InputLabelImageType::IndexType centerofstructure;

    centerofstructure[0] = region.GetIndex()[0] + (region.GetSize()[0] - 1)/2;
    centerofstructure[1] = region.GetIndex()[1] + (region.GetSize()[1] - 1)/2;
    centerofstructure[2] = region.GetIndex()[2] + (region.GetSize()[2] - 1)/2;

    lbit.SetLocation(centerofstructure);


    it.GoToBegin();
    lbout.GoToBegin();

    while(!it.IsAtEnd()){
        if(it.InBounds()){
            double background = 0;
            double foreground = 0;
            for(int i = 0; i < it.Size(); i++){
                if(lbit.GetPixel(i) == 2){
                    foreground += it.GetPixel(i);
                }else{
                    background += it.GetPixel(i);
                }
            }
            
            if(foreground < 800 && background > 8000){
                for(int i = 0; i < it.Size(); i++){
                        lbout.SetPixel(i, 1);
                }
            }
        }
        ++it;
        ++lbout;
    }


    typedef itk::ImageFileWriter< InputLabelImageType > InputLabelImageFileWriterType;
    InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(outputimg);
    writer->Update();


    return EXIT_SUCCESS;
}
