
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
#include <itkLabelGeometryImageFilter.h>
#include <itkSignedDanielssonDistanceMapImageFilter.h>
#include <vnl/vnl_cross.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>
#include "best_ellipse_fit.h"
#include "best_ellipse_fit_t.h"

using namespace std;


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

typedef itk::LabelStatisticsImageFilter< InputImageType, InputImageType > LabelStatisticsImageFilterType;
typedef LabelStatisticsImageFilterType::Pointer LabelStatisticsImageFilterPointerType;

typedef itk::NearestNeighborInterpolateImageFunction<InputImageType> InterpolateImageType;
typedef typename InterpolateImageType::Pointer InterpolateImagePointerType;

typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;

typedef itk::LabelGeometryImageFilter< InputImageType > LabelGeometryImageFilterType;

typedef itk::Image< float, dimension> OutputImageSignedDistanceType;
typedef itk::ImageRegionIterator< OutputImageSignedDistanceType > DistanceRegionIteratorType;

typedef itk::SignedDanielssonDistanceMapImageFilter<InputImageType, OutputImageSignedDistanceType, OutputImageSignedDistanceType > SignedDanielssonDistanceMapImageFilterType;


vector< vnl_vector<double> > GetBoundaryPoints(InputImagePointerType labelimage){
    vector< vnl_vector<double> > boundary_points; 

    SignedDanielssonDistanceMapImageFilterType::Pointer distance_filter = SignedDanielssonDistanceMapImageFilterType::New();
    distance_filter->SetInput(labelimage);
    distance_filter->Update();
    OutputImageSignedDistanceType::Pointer distance_image = distance_filter->GetOutput();

    DistanceRegionIteratorType distance_it = DistanceRegionIteratorType(distance_image, distance_image->GetLargestPossibleRegion());

    while(!distance_it.IsAtEnd()){

        if(round(distance_it.Get()) == 0){
            vnl_vector<double> boundary_point(dimension);
            for(int i = 0; i < dimension; i++){
                boundary_point[i] = distance_it.GetIndex()[i];    
            }
            boundary_points.push_back(boundary_point);
        }

        ++distance_it;
    }

    return boundary_points;
}

string json_vector_str(vnl_vector<double> vect){
    ostringstream os_vect;
    os_vect<<"[";

    for(int i = 0; i < vect.size() - 1; i++){
        os_vect<<vect[i];
        os_vect<<",";
    }
    os_vect<<vect[vect.size() - 1]<<"]";
    return os_vect.str();
}

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputLabelFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }
    cout << "The input label image is: " << inputLabelFilename << endl;

    
    
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

    typedef itk::ImageRegionIterator<InputImageType> InputImageIteratorType;
    InputImageIteratorType itout = InputImageIteratorType(outimg, outimg->GetLargestPossibleRegion());

    if(caliperMode){
        cout << "Fitting using calipers..."<<endl;

        typedef itk::ConnectedComponentImageFilter <InputImageType, InputImageType > ConnectedComponentImageFilterType;

        ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New ();
        connected->SetInput(labelimage);
        connected->Update();

        LabelStatisticsImageFilterPointerType labelstats = LabelStatisticsImageFilterType::New();

        labelstats->SetInput(connected->GetOutput());
        labelstats->SetLabelInput(connected->GetOutput());

        labelstats->Update();

        
        cout << "Number of labels: " << labelstats->GetNumberOfLabels() << endl;
        cout << endl;

        typedef itk::ContinuousIndex<double, dimension> ContinuousIndexType;
        

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
                    if(circleMode){
                        ab[1] = ab[0];
                    }else{
                        ab[1] = ab[0]*majorRadiusFactor;
                    }
                    

                    v1 = v1.normalize();

                    vnl_vector<double> v2(3, 0);
                    v2[0] = -1;

                    // (x-h)²/a²+(y-k)²/b²=1.
                    vnl_matrix<double> transform(2,2);
                    transform.set_identity();
                    transform[0][0] = dot_product(v1, v2);
                    transform[0][1] = -vnl_cross_3d(v1, v2).magnitude();
                    transform[1][0] = vnl_cross_3d(v1, v2).magnitude();
                    transform[1][1] = dot_product(v1, v2);

                    vnl_matrix<double> inverse_transform = vnl_matrix_inverse<double>(transform);
                    
                    cout<<ab<<endl;

                    itout.GoToBegin();

                    while(!itout.IsAtEnd()){
                    
                        IndexType xy_index = itout.GetIndex();
                        vnl_vector<double> xy(3, 1);
                        xy[0] = xy_index[0];
                        xy[1] = xy_index[1];

                        xy -= center_vect;
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
                }
            }
        }

        cout<<"Writing: "<<outputFilename<<endl;

        InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();
        writer->SetFileName(outputFilename);
        writer->SetInput(outimg);
        writer->Update();

    }else{

        LabelGeometryImageFilterType::Pointer labelGeometryImageFilter = LabelGeometryImageFilterType::New();
        labelGeometryImageFilter->SetInput( labelimage );
        labelGeometryImageFilter->SetIntensityInput( labelimage );

        labelGeometryImageFilter->CalculatePixelIndicesOn();
        labelGeometryImageFilter->CalculateOrientedBoundingBoxOn();
        labelGeometryImageFilter->CalculateOrientedLabelRegionsOn();
        labelGeometryImageFilter->CalculateOrientedIntensityRegionsOn();

        labelGeometryImageFilter->Update();

        LabelGeometryImageFilterType::LabelsType::iterator liit;
        LabelGeometryImageFilterType::LabelsType allLabels = labelGeometryImageFilter->GetLabels();

        for(liit = allLabels.begin(); liit != allLabels.end(); ++liit){
            
            LabelPixelType labelValue = *liit;
            cout<<"Label Value: "<<(int)labelValue<<endl;
            if(labelValue != 0){

                vector< vnl_vector<double> > boundary_points = GetBoundaryPoints(labelimage);

                // (x-h)²/a²+(y-k)²/b²=1.
                vnl_vector<double> x_angle_radius_center(5);
                x_angle_radius_center[0] = 0;

                x_angle_radius_center[1] = labelGeometryImageFilter->GetMajorAxisLength(labelValue)/2.0;
                x_angle_radius_center[2] = labelGeometryImageFilter->GetMajorAxisLength(labelValue)/2.0*.8;

                x_angle_radius_center[3] = labelGeometryImageFilter->GetCentroid(labelValue)[0];
                x_angle_radius_center[4] = labelGeometryImageFilter->GetCentroid(labelValue)[1];

                
                BestEllipseFitT bestfit = BestEllipseFitT(5, boundary_points.size());
                bestfit.SetBoundaryPoints(boundary_points);

                vnl_levenberg_marquardt levenberg(bestfit);

                cout<<"Initial guess: "<<x_angle_radius_center<<endl;
                levenberg.minimize(x_angle_radius_center);
                cout<<"Best fit: "<<x_angle_radius_center<<endl;

                double angle = x_angle_radius_center[0];

                vnl_vector<double> radius(2);
                radius[0] = x_angle_radius_center[1];
                radius[1] = x_angle_radius_center[2];

                vnl_vector<double> center(2);
                center[0] = x_angle_radius_center[3];
                center[1] = x_angle_radius_center[4];

                itout.GoToBegin();

                vnl_matrix<double> transform(2,2);
                transform.set_identity();
                transform[0][0] = cos(angle);
                transform[1][0] = -sin(angle);
                transform[0][1] = sin(angle);
                transform[1][1] = cos(angle);    

                while(!itout.IsAtEnd()){
                
                    IndexType xy_index = itout.GetIndex();
                    vnl_vector<double> xy(2, 0);
                    xy[0] = xy_index[0];
                    xy[1] = xy_index[1];
                    xy = xy - center;
                    xy = xy*transform;

                    double xha = pow(xy[0], 2)/(pow(radius[0], 2) + 1e-7);
                    double ykb = pow(xy[1], 2)/(pow(radius[1], 2) + 1e-7);
                    
                    itout.Set(0);
                    
                    if(xha + ykb <= 1.0){
                        itout.Set(1);
                    }

                    ++itout;
                }


                cout<<"Writing: "<<outputFilename<<endl;

                InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();
                writer->SetFileName(outputFilename + ".nrrd");
                writer->SetInput(outimg);
                writer->Update();

                ostringstream os_json;

                os_json<<"{";
                os_json<<"\"angle\":"<<angle<<",";
                os_json<<"\"radius\":"<<json_vector_str(radius)<<",";
                os_json<<"\"center\":"<<json_vector_str(center);
                os_json<<"}";

                cout<<os_json.str();

                ofstream outjson((outputFilename + ".json").c_str());
                if(outjson.is_open()){
                    outjson << os_json.str();
                    outjson.close();
                }else{
                    cerr<<"Could not create file: "<<outputFilename<<endl;
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
