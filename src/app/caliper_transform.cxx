
#include "caliper_transformCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkAffineTransform.h>
#include <itkTransformFileWriter.h>

#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkContinuousIndex.h>


#include <vnl/vnl_cross.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>


using namespace std;


//Read Image
typedef unsigned short InputPixelType;
static const int dimension = 2;
typedef itk::Image< InputPixelType, dimension> InputImageType;
typedef InputImageType::Pointer InputImagePointerType;
typedef InputImageType::IndexType IndexType;
typedef itk::ContinuousIndex<double, dimension> ContinuousIndexType;

typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;

typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
typedef InputImageFileReaderType::Pointer InputImageFileReaderPointerType;

typedef itk::ImageFileWriter< InputImageType > InputLabelImageFileWriterType;

typedef itk::LabelStatisticsImageFilter< InputImageType, InputImageType > LabelStatisticsImageFilterType;
typedef LabelStatisticsImageFilterType::Pointer LabelStatisticsImageFilterPointerType;

typedef itk::BSplineInterpolateImageFunction<InputImageType, double, double> InterpolateImageType;
typedef typename InterpolateImageType::Pointer InterpolateImagePointerType;

typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;

typedef itk::ResampleImageFilter<InputImageType, InputImageType > ResampleImageFilterType;
typedef ResampleImageFilterType::Pointer ResampleImageFilterPointerType;
typedef itk::AffineTransform< double, dimension > TransformType;
typedef TransformType::Pointer TransformPointerType;

typedef itk::TransformFileWriterTemplate< double > TransformWriterType;
typedef TransformWriterType::Pointer TransformWriterPointerType;


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


    if(inputImageFilename.compare("") == 0 || inputLabelFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;
    cout << "The input label image is: " << inputLabelFilename << endl;

    InputImageFileReaderPointerType reader = InputImageFileReaderType::New();
    reader->SetFileName(inputImageFilename);
    reader->Update();

    InputImagePointerType image = reader->GetOutput();
    
    reader = InputImageFileReaderType::New();
    reader->SetFileName(inputLabelFilename);
    reader->Update();

    InputImagePointerType labelimage = reader->GetOutput();

    // InputImagePointerType outimg = InputImageType::New();
    // outimg->SetRegions(labelimage->GetLargestPossibleRegion());
    // outimg->SetSpacing(labelimage->GetSpacing());
    // outimg->SetOrigin(labelimage->GetOrigin());
    // outimg->SetDirection(labelimage->GetDirection());
    // outimg->Allocate();
    // outimg->FillBuffer(0);

    // typedef itk::ImageRegionIterator<InputImageType> InputImageIteratorType;
    // InputImageIteratorType itout = InputImageIteratorType(outimg, outimg->GetLargestPossibleRegion());

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

                ContinuousIndexType start_index = regioni.GetIndex();
                start_index[0] = start_index[0] + regioni.GetSize()[0]/2.0;
                start_index[1] = start_index[1] + regioni.GetSize()[1]/2.0;

                ContinuousIndexType end_index = regionj.GetIndex();
                end_index[0] = end_index[0] + regionj.GetSize()[0]/2.0;
                end_index[1] = end_index[1] + regionj.GetSize()[1]/2.0;

                ContinuousIndexType center_index;
                center_index[0] = (image->GetLargestPossibleRegion().GetSize()[0])/2.0;
                center_index[1] = (image->GetLargestPossibleRegion().GetSize()[1])/2.0;

                InputImageType::PointType start_point;
                InputImageType::PointType end_point;
                InputImageType::PointType middle_point;
                InputImageType::PointType center_point;

                labelimage->TransformContinuousIndexToPhysicalPoint(start_index, start_point);
                labelimage->TransformContinuousIndexToPhysicalPoint(end_index, end_point);
                labelimage->TransformContinuousIndexToPhysicalPoint(center_index, center_point);

                middle_point[0] = (start_point[0] + end_point[0])/2.0;
                middle_point[1] = (start_point[1] + end_point[1])/2.0;
                
                // vnl_matrix<double> transform(2,2);
                // transform.set_identity();
                // transform[0][0] = dot_product(v1, v2);
                // transform[0][1] = -vnl_cross_3d(v1, v2).magnitude();
                // transform[1][0] = vnl_cross_3d(v1, v2).magnitude();
                // transform[1][1] = dot_product(v1, v2);

                TransformPointerType transform = TransformType::New();

                // TransformType::OutputVectorType translation_origin;
                // translation_origin[0] = -middle_point[0];
                // translation_origin[1] = -middle_point[1];

                // transform->Translate( translation_origin );

                TransformType::OutputVectorType translation_center;
                translation_center[0] = middle_point[0] - center_point[0];
                translation_center[1] = middle_point[1] - center_point[1];

                vnl_vector<double> v1(2);
                v1[0] = (end_point[0] - start_point[0]);
                v1[1] = (end_point[1] - start_point[1]);
                
                double scale = 1.0/v1.magnitude() * image->GetLargestPossibleRegion().GetSize()[0] * widthRatio;
                cout<<"Scale: "<<scale<<endl;
                v1 = v1.normalize();

                vnl_vector<double> v2(2, 0);
                v2[0] = 1;

                double angle = acos(dot_product(v1, v2));
                cout<<"Angle: "<<angle<<endl;

                transform->Translate(translation_center);
                transform->SetCenter(middle_point);
                transform->Rotate2D(angle);
                transform->Scale(1.0/scale);

                ResampleImageFilterPointerType image_resample =  ResampleImageFilterType::New();

                InterpolateImagePointerType interpolator = InterpolateImageType::New();
                interpolator->SetSplineOrder(3);

                image_resample->SetInterpolator(interpolator);
                image_resample->SetDefaultPixelValue(0);
                image_resample->SetOutputSpacing(image->GetSpacing());
                image_resample->SetOutputDirection(image->GetDirection());
                image_resample->SetOutputOrigin(image->GetOrigin());
                image_resample->SetSize(image->GetLargestPossibleRegion().GetSize());
                image_resample->SetInput(image);
                image_resample->SetTransform(transform);
                image_resample->Update();

                cout<<"Writing: "<<outputFilename<<endl;

                InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();
                writer->SetFileName(outputFilename + outputImageExtension);
                writer->SetInput(image_resample->GetOutput());
                writer->Update();

                TransformWriterPointerType transform_writer = TransformWriterType::New();
                transform_writer->SetInput(transform);
                transform_writer->SetFileName(outputFilename + ".txt");
                transform_writer->Update();

            }
        }
    }
    

    return EXIT_SUCCESS;
}
