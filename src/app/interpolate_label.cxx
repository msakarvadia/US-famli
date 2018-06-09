
#include "interpolate_labelCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkContinuousIndex.h>

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputLabelFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }
    cout << "The input label image is: " << inputLabelFilename << endl;

    //Read Image
    typedef unsigned short InputPixelType;
    static const int dimension = 3;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;
    
    typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
    typedef InputImageFileReaderType::Pointer InputImageFileReaderPointerType;

    typedef itk::ImageFileWriter< InputImageType > InputLabelImageFileWriterType;
    
    InputImageFileReaderPointerType reader = InputImageFileReaderType::New();
    reader->SetFileName(inputLabelFilename);
    reader->Update();

    InputImagePointerType labelimage = reader->GetOutput();

    InputImagePointerType outimg = InputImageType::New();
    outimg->SetRegions(labelimage->GetLargestPossibleRegion());
    outimg->SetSpacing(labelimage->GetSpacing());
    outimg->SetOrigin(labelimage->GetOrigin());
    outimg->SetDirection(labelimage->GetDirection());
    outimg->Allocate();
    outimg->FillBuffer(0);

    typedef itk::ConnectedComponentImageFilter <InputImageType, InputImageType > ConnectedComponentImageFilterType;

    ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New ();
    connected->SetInput(labelimage);
    connected->Update();

    typedef itk::LabelStatisticsImageFilter< InputImageType, InputImageType > LabelStatisticsImageFilterType;
    typedef LabelStatisticsImageFilterType::Pointer LabelStatisticsImageFilterPointerType;

    

    LabelStatisticsImageFilterPointerType labelstats = LabelStatisticsImageFilterType::New();

    labelstats->SetInput(connected->GetOutput());
    labelstats->SetLabelInput(connected->GetOutput());

    labelstats->Update();

    
    cout << "Number of labels: " << labelstats->GetNumberOfLabels() << endl;
    cout << endl;

    typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
    typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;

    typedef itk::ContinuousIndex<double, dimension> ContinuousIndexType;

    typedef itk::NeighborhoodIterator<InputImageType> InputImageNeighborhoodIteratorType;
    InputImageNeighborhoodIteratorType::RadiusType radius;
    radius[0] = radiusVector[0];
    radius[1] = radiusVector[1];
    radius[2] = radiusVector[2];

    InputImageNeighborhoodIteratorType itout = InputImageNeighborhoodIteratorType(radius, outimg, outimg->GetLargestPossibleRegion());

    for(ValidLabelValuesType::const_iterator liit=labelstats->GetValidLabelValues().begin(); liit != labelstats->GetValidLabelValues().end(); ++liit){
    
        if ( labelstats->HasLabel(*liit) ){
            LabelPixelType labelValue = *liit;
            std::cout << "min: " << labelstats->GetMinimum( labelValue ) << std::endl;
            std::cout << "max: " << labelstats->GetMaximum( labelValue ) << std::endl;
            std::cout << "median: " << labelstats->GetMedian( labelValue ) << std::endl;
            std::cout << "mean: " << labelstats->GetMean( labelValue ) << std::endl;
            std::cout << "sigma: " << labelstats->GetSigma( labelValue ) << std::endl;
            std::cout << "variance: " << labelstats->GetVariance( labelValue ) << std::endl;
            std::cout << "sum: " << labelstats->GetSum( labelValue ) << std::endl;
            std::cout << "count: " << labelstats->GetCount( labelValue ) << std::endl;
            // std::cout << "box: " << labelstats->GetBoundingBox( labelValue ) << std::endl; // can't output a box
            std::cout << "region: " << labelstats->GetRegion( labelValue ) << std::endl;
            std::cout << std::endl << std::endl;
        }

        for(ValidLabelValuesType::const_iterator ljit = liit + 1; ljit != labelstats->GetValidLabelValues().end(); ++ljit){
            if(*liit != *ljit && *ljit != 0 && *liit != 0){
                InputImageType::RegionType regioni = labelstats->GetRegion( *liit );
                InputImageType::RegionType regionj = labelstats->GetRegion( *ljit );

                InputImageType::IndexType start = regioni.GetIndex();
                start[0] += regioni.GetSize()[0]/2;
                start[1] += regioni.GetSize()[1]/2;
                start[2] += regioni.GetSize()[2]/2;

                InputImageType::IndexType end = regionj.GetIndex();
                end[0] += regionj.GetSize()[0]/2;
                end[1] += regionj.GetSize()[1]/2;
                end[2] += regionj.GetSize()[2]/2;

                ContinuousIndexType esv;
                esv[0] = (end[0] - start[0]);
                esv[1] = (end[1] - start[1]);
                esv[2] = (end[2] - start[2]);

                double step = max(fabs(esv[0]), max(fabs(esv[1]), fabs(esv[2])));
                if(step > 0){
                    step = 1.0/step;
                    
                    for(double delta = 0.0; delta <= 1.0; delta += step){
                        InputImageType::IndexType cu_i;
                        cu_i[0] = start[0] + delta*esv[0];
                        cu_i[1] = start[1] + delta*esv[1];
                        cu_i[2] = start[2] + delta*esv[2];
                        
                        itout.SetLocation(cu_i);

                        if(itout.InBounds()){

                            for(int i = 0; i < itout.Size(); i++){
                                if(labelValue == -1){
                                    itout.SetPixel(i, *liit);
                                }else{
                                    itout.SetPixel(i, labelValue);
                                }
                                
                            }
                        }

                        
                    }
                }else{
                    cerr<<"Cannot continue!, step size is 0!"<<endl;
                    return EXIT_FAILURE;
                }
            }
        }
    }

    cout<<"Writing: "<<outputFilename<<endl;


    InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();
    writer->SetFileName(outputFilename);
    writer->SetInput(outimg);
    writer->Update();


    return EXIT_SUCCESS;
}
