
#include "fit_ellipseCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkContinuousIndex.h>
#include <vnl/vnl_cross.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>
#include "best_ellipse_fit.h"

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputLabelFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }
    cout << "The input label image is: " << inputLabelFilename << endl;

    //Read Image
    typedef unsigned char InputPixelType;
    static const int dimension = 3;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;
    typedef InputImageType::IndexType IndexType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;
    
    typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
    typedef InputImageFileReaderType::Pointer InputImageFileReaderPointerType;

    typedef itk::ImageFileWriter< InputImageType > InputLabelImageFileWriterType;

    InputImageFileReaderPointerType reader = InputImageFileReaderType::New();
    reader->SetFileName(inputFilename);
    reader->Update();

    InputImagePointerType image = reader->GetOutput();
    
    reader = InputImageFileReaderType::New();
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

    typedef itk::NearestNeighborInterpolateImageFunction<InputImageType> InterpolateImageType;
    typedef typename InterpolateImageType::Pointer InterpolateImagePointerType;

    LabelStatisticsImageFilterPointerType labelstats = LabelStatisticsImageFilterType::New();

    labelstats->SetInput(connected->GetOutput());
    labelstats->SetLabelInput(connected->GetOutput());

    labelstats->Update();

    
    cout << "Number of labels: " << labelstats->GetNumberOfLabels() << endl;
    cout << endl;

    typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
    typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;

    typedef itk::ContinuousIndex<double, dimension> ContinuousIndexType;

    // typedef itk::NeighborhoodIterator<InputImageType> InputImageNeighborhoodIteratorType;
    // InputImageNeighborhoodIteratorType::RadiusType radius;
    // radius[0] = radiusVector[0];
    // radius[1] = radiusVector[1];
    // radius[2] = radiusVector[2];

    typedef itk::ImageRegionIterator<InputImageType> InputImageIteratorType;
    InputImageIteratorType itout = InputImageIteratorType(outimg, outimg->GetLargestPossibleRegion());

    if(labelstats->GetNumberOfLabels() > 3){
        cerr<<"More than 3 regions found"<<endl;
        return 1;
    }

    for(ValidLabelValuesType::const_iterator liit=labelstats->GetValidLabelValues().begin(); liit != labelstats->GetValidLabelValues().end() - 1; ++liit){
    
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
                start[0] += regioni.GetSize()[0]/2.0;
                start[1] += regioni.GetSize()[1]/2.0;
                start[2] += regioni.GetSize()[2]/2.0;

                InputImageType::IndexType end = regionj.GetIndex();
                end[0] += regionj.GetSize()[0]/2.0;
                end[1] += regionj.GetSize()[1]/2.0;
                end[2] += regionj.GetSize()[2]/2.0;

                // Center of ellipse
                vnl_vector<double> center_vect(3);
                center_vect[0] = (end[0] + start[0])/2.0;
                center_vect[1] = (end[1] + start[1])/2.0;
                center_vect[2] = (end[2] + start[2])/2.0;
                

                vnl_vector<double> v1(3);
                v1[0] = start[0] - end[0];
                v1[1] = start[1] - end[1];
                v1[2] = start[2] - end[2];
                v1 /= 2.0;
                
                //find radius a this is the initial guess
                vnl_vector<double> ab(2);
                ab[0] = v1.magnitude();
                ab[1] = ab[0]*1.2;

                v1 = v1.normalize();

                vnl_vector<double> v2(3, 0);
                v2[0] = -1;

                // (x-h)²/a²+(y-k)²/b²=1.
                vnl_matrix<double> transform(3,3);
                transform.set_identity();
                transform[0][0] = dot_product(v1, v2);
                transform[0][1] = -vnl_cross_3d(v1, v2).magnitude();
                transform[0][2] = center_vect[0];
                transform[1][0] = vnl_cross_3d(v1, v2).magnitude();
                transform[1][1] = dot_product(v1, v2);
                transform[1][2] = center_vect[1];

                vnl_matrix<double> inverse_transform = vnl_matrix_inverse<double>(transform);

                int samples = 100;
                BestEllipseFit bestfit = BestEllipseFit(samples);
                bestfit.SetInput(image);
                bestfit.SetTransform(transform);

                vnl_levenberg_marquardt levenberg(bestfit);

                cout<<ab<<endl;
                levenberg.minimize(ab);
                cout<<ab<<endl;
                InterpolateImagePointerType interpolate = InterpolateImageType::New();
                interpolate->SetInputImage(image);


                itout.GoToBegin();

                while(!itout.IsAtEnd()){
                
                    IndexType xy_index = itout.GetIndex();
                    vnl_vector<double> xy(3, 1);
                    xy[0] = xy_index[0];
                    xy[1] = xy_index[1];

                    xy = inverse_transform*xy;
                    // vnl_vector<double> xy(3, 1);
                    // xy[0] = start_x * (1.0 - ratio) + end_x * ratio;

                    double xha = xy[0]*xy[0]/(ab[0]*ab[0]);
                    double y_e = sqrt((1.0 - xha)*(ab[1]*ab[1]));
                    
                    if(!isnan(y_e) && -y_e <= xy[1] && xy[1] <= y_e){
                        itout.Set(1);
                    }

                    ++itout;
                }

                // double start_x = -ab[0];
                // double end_x = ab[0];

                // // y = sqrt((1 - (x-h)²/a²) * b²) + k
                // for(int sample = 0; sample < samples; sample++){
                    
                //     double ratio = (double)sample/(samples - 1);
                    
                //     {
                //         vnl_vector<double> xy(3, 1);
                //         xy[0] = start_x * (1.0 - ratio) + end_x * ratio;

                //         double xha = xy[0]*xy[0]/(ab[0]*ab[0]);
                //         xy[1] = sqrt((1.0 - xha)*(ab[1]*ab[1]));
                        
                //         if(!isnan(xy[0]) && !isnan(xy[1])){

                //             xy = transform*xy;

                //             IndexType xy_index;
                //             xy_index[0] = round(xy[0]);
                //             xy_index[1] = round(xy[1]);
                //             xy_index[2] = 0;

                //             InputPixelType pix = 0;
                            
                //             if(interpolate->IsInsideBuffer(xy_index)){
                //                 outimg->SetPixel(xy_index, 1);
                //             }
                //         }
                //     }
                //     {
                //         vnl_vector<double> xy(3, 1);
                //         xy[0] = start_x * (1.0 - ratio) + end_x * ratio;

                //         double xha = xy[0]*xy[0]/(ab[0]*ab[0]);
                //         xy[1] = -sqrt((1.0 - xha)*(ab[1]*ab[1]));
                        
                //         if(!isnan(xy[0]) && !isnan(xy[1])){

                //             xy = transform*xy;

                //             IndexType xy_index;
                //             xy_index[0] = round(xy[0]);
                //             xy_index[1] = round(xy[1]);
                //             xy_index[2] = 0;

                //             InputPixelType pix = 0;
                            
                //             if(interpolate->IsInsideBuffer(xy_index)){
                //                 outimg->SetPixel(xy_index, 1);
                //             }
                //         }
                //     }
                // }
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
