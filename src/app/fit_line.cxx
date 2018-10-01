
#include "fit_lineCLP.h"

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
#include <itkNumericTraits.h>
#include <vnl/vnl_cross.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>
#include "best_line_fit.h"

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

    InterpolateImagePointerType interpolate = InterpolateImageType::New();
    interpolate->SetInputImage(outimg);

    typedef itk::ImageRegionIterator<InputImageType> InputImageIteratorType;
    InputImageIteratorType itout = InputImageIteratorType(outimg, outimg->GetLargestPossibleRegion());
    InputImageIteratorType itlab = InputImageIteratorType(labelimage, labelimage->GetLargestPossibleRegion());

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
            
            LabelGeometryImageFilterType::LabelPointType label_bb_origin = labelGeometryImageFilter->GetOrientedBoundingBoxOrigin(labelValue);
            LabelGeometryImageFilterType::LabelPointType label_bb_size = labelGeometryImageFilter->GetOrientedBoundingBoxSize(labelValue);
            vector< vnl_vector<double> > boundary_points = GetBoundaryPoints(labelimage);

            // (x-h)²/a²+(y-k)²/b²=1.
            vnl_vector<double> x_lineparam(4);

            x_lineparam[0] = label_bb_origin[0];
            x_lineparam[1] = label_bb_origin[1];
            x_lineparam[2] = label_bb_size[0] + label_bb_origin[0];
            x_lineparam[3] = label_bb_size[1] + label_bb_origin[1];
            
            BestLineFit bestfit = BestLineFit(4, boundary_points.size());
            bestfit.SetPoints(boundary_points);

            vnl_levenberg_marquardt levenberg(bestfit);

            cout<<"Initial guess: "<<x_lineparam<<endl;
            levenberg.minimize(x_lineparam);
            cout<<"Best fit: "<<x_lineparam<<endl;

            vnl_vector< double > p1(x_lineparam.data_block(), 2);
            vnl_vector< double > p2(x_lineparam.data_block() + 2, 2);

            vnl_vector<double> min_point(2, itk::NumericTraits< double >::max());
            vnl_vector<double> max_point(2, itk::NumericTraits< double >::min());
            
            itlab.GoToBegin();

            while(!itlab.IsAtEnd()){

                if(itlab.Get()){

                    IndexType xy_index = itlab.GetIndex();

                    vnl_vector<double> p0(2, 0);
                    p0[0] = xy_index[0];
                    p0[1] = xy_index[1];

                    vnl_vector<double> v = p2 - p1;
                    vnl_vector<double> s = p0 - p1;

                    double angle = acos(dot_product(v, s)/(v.magnitude()*s.magnitude()));

                    vnl_vector<double> p0proj = p1 + v*(cos(angle)*s.magnitude()/v.magnitude() + 1e-7);
                    // cout<<p0proj<<endl;
                    
                    // double mindistance = fabs((p2[1] - p1[1])*p0[0] - (p2[0] - p1[0])*p0[1] + p2[0]*p1[1] - p2[1]*p1[0])/(sqrt(pow(p2[1] - p1[1], 2) + (pow(p2[0] - p1[0], 2))) + 1e-7);
                    // double c_angle = asin(mindistance/(p1 - p0).magnitude());

                    // vnl_matrix<double> transform(2,2);
                    // transform.set_identity();
                    // transform[0][0] = cos(c_angle);
                    // transform[1][0] = -sin(c_angle);
                    // transform[0][1] = sin(c_angle);
                    // transform[1][1] = cos(c_angle);

                    // p0 = p0*transform;

                    xy_index[0] = round(p0proj[0]);
                    xy_index[1] = round(p0proj[1]);

                    if(interpolate->IsInsideBuffer(xy_index)){
                        outimg->SetPixel(xy_index, 1);    
                        for(int i = 0; i < p0proj.size(); i++){
                            min_point[i] = min((double)xy_index[i], min_point[i]);
                            max_point[i] = max((double)xy_index[i], max_point[i]);
                        }
                    }
                }
                
                ++itlab;
            }

            cout<<"Writing: "<<outputFilename<<endl;

            InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();
            writer->SetFileName(outputFilename + ".nrrd");
            writer->SetInput(outimg);
            writer->Update();

            ostringstream os_json;

            os_json<<"{";
            os_json<<"\"min\":"<<json_vector_str(min_point)<<",";
            os_json<<"\"max\":"<<json_vector_str(max_point);
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

    return EXIT_SUCCESS;
}
