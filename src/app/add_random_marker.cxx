
#include "add_random_markerCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkLabelImageToLabelMapFilter.h>
#include <itkLabelMapMaskImageFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkNeighborhoodIterator.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkImageRandomConstIteratorWithIndex.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkImageDuplicator.h>
#include <vnl/vnl_sample.h>


using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;
    if(inputLabelFilename.compare("") != 0){
        cout << "The input label image is: " << inputLabelFilename << endl;
    }

    //Read Image
    typedef unsigned char InputPixelType;
    static const int dimension = 2;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;
    InputImagePointerType imgin;

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
    
    typedef itk::ImageDuplicator< InputImageType > InputLabelImageDuplicatorType;
    typedef InputLabelImageDuplicatorType::Pointer InputLabelImageDuplicatorPointerType;

    InputLabelImageDuplicatorPointerType duplicator = InputLabelImageDuplicatorType::New();
    duplicator->SetInputImage(imgin);
    duplicator->Update();

    InputImagePointerType outputimg = duplicator->GetOutput();
    
    typedef unsigned char InputLabelPixelType;
    typedef itk::Image< InputLabelPixelType, dimension> InputLabelImageType;
    typedef InputLabelImageType::Pointer InputLabelImagePointerType;

    typedef itk::ImageFileReader< InputLabelImageType > InputLabelImageFileReaderType;
    typedef InputLabelImageFileReaderType::Pointer InputImageLabelFileReaderPointerType;


    InputLabelImagePointerType maskimage;

    if(inputMaskFilename.compare("") != 0){
        InputImageLabelFileReaderPointerType readerlm = InputLabelImageFileReaderType::New();
        readerlm->SetFileName(inputMaskFilename);
        readerlm->Update();
        maskimage = readerlm->GetOutput();
    }


    typedef itk::NeighborhoodIterator<InputImageType> InputImageNeighborhoodIteratorType;
    typedef itk::NeighborhoodIterator<InputLabelImageType> InputLabelImageNeighborhoodIteratorType;
    typedef itk::ImageRandomConstIteratorWithIndex< InputImageType > RandomConstImageRegionIteratorType;

    InputLabelImagePointerType labelimage;

    if(inputLabelFilename.compare("") != 0){
        InputImageLabelFileReaderPointerType readerlm = InputLabelImageFileReaderType::New();
        readerlm->SetFileName(inputLabelFilename);
        readerlm->Update();

        labelimage = readerlm->GetOutput();

        typedef itk::ConnectedComponentImageFilter <InputImageType, InputImageType > ConnectedComponentImageFilterType;

        ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New ();
        connected->SetInput(labelimage);
        connected->Update();

        typedef itk::LabelStatisticsImageFilter< InputImageType, InputLabelImageType > LabelStatisticsImageFilterType;
        typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;
        typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
        LabelStatisticsImageFilterType::Pointer labelStatisticsImageFilter = LabelStatisticsImageFilterType::New();
        labelStatisticsImageFilter->SetLabelInput( connected->GetOutput() );
        labelStatisticsImageFilter->SetInput(imgin);
        labelStatisticsImageFilter->Update();

        for(ValidLabelValuesType::const_iterator liit=labelStatisticsImageFilter->GetValidLabelValues().begin() + 1; liit != labelStatisticsImageFilter->GetValidLabelValues().end(); ++liit){
            InputImageType::RegionType region =  labelStatisticsImageFilter->GetRegion( *liit );

            std::cout << "region: " << region << std::endl;

            InputImageType::SizeType radius;

            radius[0] = (region.GetSize()[0] - 1)/2;
            radius[1] = (region.GetSize()[1] - 1)/2;
            // radius[2] = (region.GetSize()[2] - 1)/2;
            

            InputImageNeighborhoodIteratorType it(radius, imgin, imgin->GetLargestPossibleRegion());
            InputImageNeighborhoodIteratorType itout(radius, outputimg, outputimg->GetLargestPossibleRegion());
            RandomConstImageRegionIteratorType itoutrand = RandomConstImageRegionIteratorType(outputimg, outputimg->GetLargestPossibleRegion());
            itoutrand.SetNumberOfSamples(numSamples);

            InputImageType::IndexType centerofstructure;

            centerofstructure[0] = region.GetIndex()[0] + (region.GetSize()[0] - 1)/2;
            centerofstructure[1] = region.GetIndex()[1] + (region.GetSize()[1] - 1)/2;
            // centerofstructure[2] = region.GetIndex()[2] + (region.GetSize()[2] - 1)/2;

            it.SetLocation(centerofstructure);

            itoutrand.GoToBegin();

            while(!itoutrand.IsAtEnd()){
                itout.SetLocation(itoutrand.GetIndex());

                if(it.InBounds() && itout.InBounds() && (!maskimage || maskimage->GetPixel(itoutrand.GetIndex()) != 0)){
                    for(int i = 0; i < it.Size(); i++){
                        double pix = fabs(itout.GetPixel(i)*0.25 + it.GetPixel(i)*0.75);
                        itout.SetPixel(i, (InputPixelType) pix);
                    }
                }
                ++itoutrand;
            }
        }
    }


    if(numSamplesCross > 0){
        InputImageType::SizeType radius;

        radius[0] = 4;
        radius[1] = 4;
        // radius[2] = 0;

        InputImageNeighborhoodIteratorType itout(radius, outputimg, outputimg->GetLargestPossibleRegion());

        RandomConstImageRegionIteratorType itoutrand = RandomConstImageRegionIteratorType(outputimg, outputimg->GetLargestPossibleRegion());
        itoutrand.SetNumberOfSamples(numSamplesCross);

        itoutrand.GoToBegin();

        vector< InputImageNeighborhoodIteratorType::OffsetType > offsets;
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1,0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1,0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,-1,0});

        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,-1});
        

        while(!itoutrand.IsAtEnd()){
            itout.SetLocation(itoutrand.GetIndex());

            if(itout.InBounds() && (!maskimage || maskimage->GetPixel(itoutrand.GetIndex()) != 0)){
                for(int i = 0; i < itout.Size(); i++){
                    itout.SetPixel(i, abs(vnl_sample_uniform(itout.GetPixel(i) - 50, itout.GetPixel(i) + 50)));
                }
                for(int i = 0; i < offsets.size(); i++){
                    itout.SetPixel(offsets[i], 255);
                }
            }
            ++itoutrand;
        }
    }

    if(numSamplesCrossBig > 0){
        InputImageType::SizeType radius;

        radius[0] = 10;
        radius[1] = 10;
        // radius[2] = 0;

        InputImageNeighborhoodIteratorType itout(radius, outputimg, outputimg->GetLargestPossibleRegion());

        RandomConstImageRegionIteratorType itoutrand = RandomConstImageRegionIteratorType(outputimg, outputimg->GetLargestPossibleRegion());
        itoutrand.SetNumberOfSamples(numSamplesCrossBig);

        itoutrand.GoToBegin();

        vector< InputImageNeighborhoodIteratorType::OffsetType > offsets;

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-8,0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-8,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-7,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-6,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-5,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-4,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-3,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-3,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-4,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-5,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-6,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-7,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-8,-1,0});

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){8,0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){8,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){7,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){6,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){5,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){4,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){3,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){3,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){4,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){5,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){6,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){7,-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){8,-1,0});

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, -8 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -8 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -7 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -6 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -5 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -4 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -3 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -2 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, -2 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -8 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -7 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -6 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -5 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -4 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -3 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -2 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, 8 ,0});
        
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, 2 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 2 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 3 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 4 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 5 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 6 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 7 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 8 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 2 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 3 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 4 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 5 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 6 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 7 ,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 8 ,0});

        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-8,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-8,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-7,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-6,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-5,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-4,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-3,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-3,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-4,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-5,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-6,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-7,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-8,-1});

        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){8,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){8,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){7,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){6,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){5,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){4,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){3,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){3,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){4,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){5,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){6,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){7,-1});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){8,-1});

        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, -8});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -8});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -7});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -6});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -5});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -4});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -3});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, -2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, -2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -8});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -7});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -6});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -5});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -4});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -3});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, -2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, 8});
        
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, 2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 3});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 4});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 5});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 6});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 7});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1, 8});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 3});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 4});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 5});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 6});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 7});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1, 8});
        
        while(!itoutrand.IsAtEnd()){
            itout.SetLocation(itoutrand.GetIndex());

            if(itout.InBounds() && (!maskimage || maskimage->GetPixel(itoutrand.GetIndex()) != 0)){
                for(int i = 0; i < itout.Size(); i++){
                    itout.SetPixel(i, abs(vnl_sample_uniform(itout.GetPixel(i) - 10, itout.GetPixel(i) + 10)));
                }
                for(int i = 0; i < offsets.size(); i++){
                    itout.SetPixel(offsets[i], abs(vnl_sample_uniform(0, 10)));
                }
            }
            ++itoutrand;
        }
    }

    if(numSamplesCrossUNC > 0){
        cout<<"UNC cross small"<<endl;
        InputImageType::SizeType radius;

        radius[0] = 1;
        radius[1] = 1;
        // radius[2] = 0;

        InputImageNeighborhoodIteratorType itout(radius, outputimg, outputimg->GetLargestPossibleRegion());

        RandomConstImageRegionIteratorType itoutrand = RandomConstImageRegionIteratorType(outputimg, outputimg->GetLargestPossibleRegion());
        itoutrand.SetNumberOfSamples(numSamplesCrossUNC);

        itoutrand.GoToBegin();

        // vector< InputImageNeighborhoodIteratorType::OffsetType > offsets;

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,-2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1,-2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,-2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1,-2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,-2});

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,-1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1,-1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,-1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1,-1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,-1});

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1,0});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,0});

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1,1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1,1});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,1});

        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){1,2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0,2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-1,2});
        // offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,2});
        

        while(!itoutrand.IsAtEnd()){
            itout.SetLocation(itoutrand.GetIndex());

            if(itout.InBounds() && (!maskimage || maskimage->GetPixel(itoutrand.GetIndex()) != 0)){
                for(int i = 0; i < itout.Size(); i++){
                    itout.SetPixel(i, abs(vnl_sample_uniform(itout.GetPixel(i) - 50, itout.GetPixel(i) + 50)));
                }
                // for(int i = 0; i < offsets.size(); i++){
                //     itout.SetPixel(offsets[i], 255);
                // }
            }
            ++itoutrand;
        }
    }

    if(numSamplesCrossBigUNC > 0){
        InputImageType::SizeType radius;

        radius[0] = 5;
        radius[1] = 5;
        // radius[2] = 0;

        InputImageNeighborhoodIteratorType itout(radius, outputimg, outputimg->GetLargestPossibleRegion());

        RandomConstImageRegionIteratorType itoutrand = RandomConstImageRegionIteratorType(outputimg, outputimg->GetLargestPossibleRegion());
        itoutrand.SetNumberOfSamples(numSamplesCrossBigUNC);

        itoutrand.GoToBegin();

        vector< InputImageNeighborhoodIteratorType::OffsetType > offsets;
        
        
    
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-4,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-3,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){-2,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){2,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){3,0});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){4,0});
        
        
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, -4});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, -3});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, -2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, 2});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, 3});
        offsets.push_back((InputImageNeighborhoodIteratorType::OffsetType){0, 4});
        
        
        
        while(!itoutrand.IsAtEnd()){
            itout.SetLocation(itoutrand.GetIndex());

            if(itout.InBounds() && (!maskimage || maskimage->GetPixel(itoutrand.GetIndex()) != 0)){
                for(int i = 0; i < offsets.size(); i++){
                    itout.SetPixel(offsets[i], abs(vnl_sample_uniform(220, 230)));
                }
                for(int i = 0; i < itout.Size(); i++){
                    itout.SetPixel(i, abs(vnl_sample_uniform(itout.GetPixel(i) - 10, itout.GetPixel(i) + 10)));
                }
            }
            ++itoutrand;
        }
    }

    if(inputSamplesFilename.compare("") != 0 && inputSamplesLabelFilename.compare("") != 0){

        typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
        typedef InputImageFileReaderType::Pointer InputImageFileReaderPointerType;

        typedef itk::LabelStatisticsImageFilter< InputLabelImageType, InputLabelImageType > LabelStatisticsImageFilterType;
        typedef LabelStatisticsImageFilterType::Pointer LabelStatisticsImageFilterPointerType;
        typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;
        typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
        
        InputImageFileReaderPointerType reader = InputImageFileReaderType::New();
        reader->SetFileName(inputSamplesFilename.c_str());
        reader->Update();

        InputImagePointerType imgsamples = reader->GetOutput();

        InputImageLabelFileReaderPointerType readerls = InputLabelImageFileReaderType::New();
        readerls->SetFileName(inputSamplesLabelFilename);
        readerls->Update();
        InputLabelImagePointerType imgsampleslabel = readerls->GetOutput();

        typedef itk::ConnectedComponentImageFilter <InputImageType, InputImageType > ConnectedComponentImageFilterType;

        ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New ();
        connected->SetInput(imgsampleslabel);
        connected->Update();

        LabelStatisticsImageFilterPointerType labelstats = LabelStatisticsImageFilterType::New();

        labelstats->SetInput(connected->GetOutput());
        labelstats->SetLabelInput(connected->GetOutput());

        labelstats->Update();

        
        cout << "Number of labels: " << labelstats->GetNumberOfLabels() << endl;
        cout << endl;

        RandomConstImageRegionIteratorType itoutrand = RandomConstImageRegionIteratorType(outputimg, outputimg->GetLargestPossibleRegion());
        itoutrand.SetNumberOfSamples(numSamplesImageLabel);

        itoutrand.GoToBegin();


        while(numSamplesImageLabel > 0){
            
            LabelPixelType labelValue = (LabelPixelType) (round(vnl_sample_uniform(0, labelstats->GetNumberOfLabels())));

            if ( labelstats->HasLabel(labelValue) ){
                
                InputImageType::RegionType region = labelstats->GetRegion( labelValue );

                InputImageType::IndexType middle_index = region.GetIndex();
                middle_index[0] = middle_index[0] + region.GetSize()[0]/2.0;
                middle_index[1] = middle_index[1] + region.GetSize()[1]/2.0;

                InputImageType::SizeType radius;

                radius[0] = (region.GetSize()[0] - 1)/2;
                radius[1] = (region.GetSize()[1] - 1)/2;

                InputLabelImageNeighborhoodIteratorType lit(radius, imgsamples, imgsamples->GetLargestPossibleRegion());
                InputImageNeighborhoodIteratorType itout(radius, outputimg, outputimg->GetLargestPossibleRegion());
                
                lit.SetLocation(middle_index);
                
                itout.SetLocation(itoutrand.GetIndex());

                if(lit.InBounds() && itout.InBounds()){
                    for(int i = 0; i < itout.Size(); i++){

                        if(imgsampleslabel->GetPixel(lit.GetIndex(i))){
                            double pix = fabs(itout.GetPixel(i)*0.1 + lit.GetPixel(i)*0.9);
                            itout.SetPixel(i, (InputPixelType)pix);    
                        }
                    }
                }
            }

            ++itoutrand;
            --numSamplesImageLabel;
        }
    }

    typedef itk::ImageFileWriter< InputLabelImageType > InputLabelImageFileWriterType;
    InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(outputimg);
    writer->Update();


    return EXIT_SUCCESS;
}
