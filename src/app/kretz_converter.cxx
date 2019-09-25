
#include "kretz_converterCLP.h"

// #include <itkImage.h>
// #include <itkImageFileReader.h>
// #include <itkImageFileWriter.h>
// #include <itkKretzImageIO.h>
// // #include <itkRGBPixel.h>
// #include <itkCartesianToToroidalTransform.h>
// #include <itkToroidalToCartesianTransform.h>


// using namespace std;

// int main (int argc, char * argv[]){
//     PARSE_ARGS;


//     if(inputImageFilename.compare("") == 0){
//         commandLine.getOutput()->usage(commandLine);
//         return EXIT_FAILURE;
//     }

//     cout << "The input image is: " << inputImageFilename << endl;

//     //Read Image
//     // typedef itk::RGBPixel<unsigned char> InputPixelType;
//     typedef unsigned char InputPixelType;
//     static const int dimension = 3;
//     typedef itk::Image< InputPixelType, dimension> InputImageType;
//     typedef InputImageType::Pointer InputImagePointerType;

//     typedef itk::ToroidalToCartesianTransform<double, dimension> T2CTransformType;
//     typedef itk::CartesianToToroidalTransform<double, dimension> C2TTransformType;

//     typedef itk::PointSet<InputImageType::PixelType, dimension> PointSetType;
//     typedef PointSetType::PointsContainerPointer PointsContainerPointer;
//     typedef itk::BoundingBox<PointSetType::PointIdentifier, dimension, double, PointSetType::PointsContainer> BoundingBoxType;

//     typedef itk::KretzImageIO           ImageIOType;
//     ImageIOType::Pointer kretzImageIO = ImageIOType::New();
    
//     typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
//     typedef InputImageFileReaderType::Pointer InputImageFileReaderPointerType;
//     InputImageFileReaderType::Pointer reader = InputImageFileReaderType::New();
//     reader->SetFileName(inputImageFilename.c_str());
//     reader->SetImageIO( kretzImageIO );
//     reader->Update();

    
//     InputImagePointerType toroidalImage = reader->GetOutput();

//     T2CTransformType::Pointer t2c = T2CTransformType::New();
//     t2c->SetBModeRadius(kretzImageIO->GetRadiusD());
//     t2c->SetSweepRadius(kretzImageIO->GetRadiusBStart());
//     t2c->SetResolution(kretzImageIO->GetResolution());
//     t2c->SetTableTheta(kretzImageIO->m_TableAnglesTheta);
//     t2c->SetTablePhi(kretzImageIO->m_TableAnglesPhi);

//     //Find the bounds of the toroidal volume in cartesian coordinates
//     BoundingBoxType::BoundsArrayType bounds = T2CTransformType::ComputeBounds(toroidalImage.GetPointer(), t2c.GetPointer());



//     // if(IsDoppler){
//     //     reader = ReaderType::New();
//     //     reader->SetFileName( filename );
//     //     kretzImageIO = ImageIOType::New();
//     //     kretzImageIO->SetIsDoppler(IsDoppler);
//     //     reader->SetImageIO( kretzImageIO );
//     //     reader->Update();
//     //     toroidalImage = reader->GetOutput();
//     // }

//     // typedef itk::RGBPixel<unsigned char> OutputPixelType;
//     typedef unsigned char OutputPixelType;
//     typedef itk::Image< OutputPixelType, dimension> OutputImageType;

//     typedef itk::ImageFileWriter< OutputImageType > OutputImageFileWriterType;
//     OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();

//     cout<<"Writing: "<<outputImageFilename<<endl;
//     writer->SetFileName(outputImageFilename.c_str());
//     writer->SetInput(outputimg);
//     writer->Update();


//     return EXIT_SUCCESS;
// }

/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "stdlib.h"
#include <sstream>
#include <iostream>
#include <string>

#include <itkImage.h>
#include <itkResampleImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkIdentityTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkImageFileReader.h>
#include <itkCartesianToToroidalTransform.h>
#include <itkToroidalToCartesianTransform.h>
#include <itkKretzImageIO.h>
#include <itkCastImageFilter.h>
#include <itkNormalizeImageFilter.h>
#include <itkInterpolateImageFunction.h>

#include <itkPointSetToImageFilter.h>
#include <itkMeshSpatialObject.h>
#include <itkBoundingBox.h>
#include <itkRescaleIntensityImageFilter.h>


using namespace std;

const unsigned int Dimension = 3;
typedef itk::Image<unsigned char, Dimension> ImageType;
typedef itk::Image<double, Dimension> DoubleImageType;
typedef itk::ResampleImageFilter<ImageType,ImageType> ResampleImageFilterType;
typedef itk::ImageFileWriter<ImageType> ImageWriterType;
typedef itk::LinearInterpolateImageFunction<ImageType, double> LinearInterpolatorType;
typedef itk::IdentityTransform<double, Dimension> TransformType;
typedef itk::ImageFileReader<ImageType> ImageFileReaderType;
typedef itk::ToroidalToCartesianTransform<double, Dimension> T2CTransformType;
typedef itk::CartesianToToroidalTransform<double, Dimension> C2TTransformType;
typedef itk::Mesh< double, 3 > MeshType;
typedef itk::PointSet<ImageType::PixelType, Dimension> PointSetType;
typedef PointSetType::PointsContainerPointer PointsContainerPointer;
typedef itk::BoundingBox<PointSetType::PointIdentifier, Dimension, double, PointSetType::PointsContainer> BoundingBoxType;

/*
 * Used to find a binary mask of a toroidal volume in cartesian coordinates
 */
ImageType::Pointer createMaskImage(ImageType::Pointer image)
{
  ImageType::Pointer returnImage = ImageType::New();
  returnImage->SetOrigin(image->GetOrigin());
  returnImage->SetDirection(image->GetDirection());
  returnImage->SetSpacing(image->GetSpacing());
  returnImage->SetRegions(image->GetLargestPossibleRegion());
  returnImage->Allocate();
  returnImage->FillBuffer(1);

  return returnImage;
}


int execute(std::string filename, std::string filename_out, std::vector<int> size_vec,std::vector<float> resol_vec, bool flagMask, bool flagNormalise, bool IsDoppler)
{
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( filename );

  typedef itk::KretzImageIO           ImageIOType;
  ImageIOType::Pointer kretzImageIO = ImageIOType::New();
  reader->SetImageIO( kretzImageIO );
  reader->Update();

  ImageType::Pointer toroidalImage = reader->GetOutput();
  T2CTransformType::Pointer t2c = T2CTransformType::New();
  t2c->SetBModeRadius(kretzImageIO->GetRadiusD());
  t2c->SetSweepRadius(kretzImageIO->GetRadiusBStart());
  t2c->SetResolution(kretzImageIO->GetResolution());
  t2c->SetTableTheta(kretzImageIO->m_TableAnglesTheta);
  t2c->SetTablePhi(kretzImageIO->m_TableAnglesPhi);

  //Find the bounds of the toroidal volume in cartesian coordinates
  BoundingBoxType::BoundsArrayType bounds = T2CTransformType::ComputeBounds(toroidalImage.GetPointer(), t2c.GetPointer());
  if(IsDoppler){
    reader = ReaderType::New();
    reader->SetFileName( filename );
    kretzImageIO = ImageIOType::New();
    kretzImageIO->SetIsDoppler(IsDoppler);
    reader->SetImageIO( kretzImageIO );
    reader->Update();
    toroidalImage = reader->GetOutput();
  }

  std::cout << "size " << size_vec.size() << std::endl;
  std::cout << "resol " << resol_vec.size() << std::endl;

  if(size_vec[0]==0) //if the resolution is provided set the size
  {
    size_vec[0]=(int) ((bounds[1]-bounds[0])/resol_vec[0]);
    size_vec[1]=(int) ((bounds[3]-bounds[2])/resol_vec[1]);
    size_vec[2]=(int) ((bounds[5]-bounds[4])/resol_vec[2]);
  } 
  else if(resol_vec[0]==0) //if the size is provided set the resolution
  {
    resol_vec[0]=(bounds[1]-bounds[0])/(size_vec[0]-1);
    resol_vec[1]=(bounds[3]-bounds[2])/(size_vec[1]-1);
    resol_vec[2]=(bounds[5]-bounds[4])/(size_vec[2]-1);
  }
  else
  {
    std::cerr << "Error: both resolution and size provided" <<  std::endl;

    return EXIT_FAILURE;
  }
  std::cout << "resol " << resol_vec.at(0) << " " << resol_vec.at(1) << " " << resol_vec.at(2) << std::endl;
  std::cout << "size " << size_vec[0] << " " << size_vec[1] << " " << size_vec[2]  << std::endl;

  //if a mask is required create a binary image with ones everywhere 
  //and same dimensions as input file
  if(flagMask) toroidalImage = createMaskImage(toroidalImage);

  if(flagNormalise){ //if normalised images wanted outpu double image type

    typedef itk::ImageFileWriter<DoubleImageType> ImageWriterType;
    typedef itk::CastImageFilter<ImageType,DoubleImageType> CastImageFilterType;
    typedef itk::NormalizeImageFilter<DoubleImageType, DoubleImageType> NormaliseImageFilterType;

    CastImageFilterType::Pointer castImgFilter = CastImageFilterType::New();
    castImgFilter->SetInput(toroidalImage);
    castImgFilter->Update();

    DoubleImageType::Pointer castImage = castImgFilter->GetOutput();

    NormaliseImageFilterType::Pointer normaliseFilter = NormaliseImageFilterType::New();
    normaliseFilter->SetInput(castImage);
    normaliseFilter->Update();
    DoubleImageType::Pointer normImage = normaliseFilter->GetOutput();

    DoubleImageType::PointType origin;
    origin[0] = bounds[0];
    origin[1] = bounds[2];
    origin[2] = bounds[4];

    C2TTransformType::Pointer c2t = C2TTransformType::New();
    c2t->SetBModeRadius(kretzImageIO->GetRadiusD());
    c2t->SetSweepRadius(kretzImageIO->GetRadiusBStart());
    c2t->SetResolution(kretzImageIO->GetResolution());
    c2t->SetTableTheta(kretzImageIO->m_TableAnglesTheta);
    c2t->SetTablePhi(kretzImageIO->m_TableAnglesPhi);
    typedef itk::ResampleImageFilter<DoubleImageType,DoubleImageType> ResampleFilterType;

    typedef DoubleImageType::SpacingType SpacingType;
    SpacingType spacing;
    spacing[0] = resol_vec.at(0);
    spacing[1] = resol_vec.at(1);
    spacing[2] = resol_vec.at(2);

    typedef typename DoubleImageType::SizeType SizeType;
    SizeType size;
    size[0]= size_vec[0];
    size[1]= size_vec[1];
    size[2]= size_vec[2];

    typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
          resampleFilter->SetInput(normImage);
    resampleFilter->SetTransform(c2t);
    resampleFilter->SetSize(size);
    resampleFilter->SetOutputOrigin(origin);
    resampleFilter->SetOutputSpacing(spacing);
    resampleFilter->Update();
    DoubleImageType::Pointer output = resampleFilter->GetOutput(); 

    ImageWriterType::Pointer ITKImageWriter = ImageWriterType::New();

    cout<<"Writing:"<<filename_out<<endl;
    ITKImageWriter->SetFileName(filename_out.c_str());
    ITKImageWriter->SetInput(output);
    ITKImageWriter->Update();

  } 
  else //If non normalised images required can revert to original image type
  {
    ImageType::PointType origin;
    origin[0] = bounds[0];
    origin[1] = bounds[2];
    origin[2] = bounds[4];

    typedef itk::ImageFileWriter<ImageType> ImageWriterType;
    C2TTransformType::Pointer c2t = C2TTransformType::New();
    c2t->SetBModeRadius(kretzImageIO->GetRadiusD());
    c2t->SetSweepRadius(kretzImageIO->GetRadiusBStart());
    c2t->SetResolution(kretzImageIO->GetResolution());
    c2t->SetTableTheta(kretzImageIO->m_TableAnglesTheta);
    c2t->SetTablePhi(kretzImageIO->m_TableAnglesPhi);
    typedef itk::ResampleImageFilter<ImageType,ImageType> ResampleFilterType;

    typedef ImageType::SpacingType SpacingType;
    SpacingType spacing;
    spacing[0] = resol_vec[0];
    spacing[1] = resol_vec[1];
    spacing[2] = resol_vec[2];

    typedef typename ImageType::SizeType SizeType;
    SizeType size;
    size[0]= size_vec[0];
    size[1]= size_vec[1];
    size[2]= size_vec[2];


    typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
    resampleFilter->SetInput(toroidalImage);

    resampleFilter->SetTransform(c2t);
    resampleFilter->SetSize(size);
    resampleFilter->SetOutputOrigin(origin);
    resampleFilter->SetOutputSpacing(spacing);
    resampleFilter->Update();
    ImageType::Pointer output = resampleFilter->GetOutput(); 

    ImageWriterType::Pointer ITKImageWriter = ImageWriterType::New();

    cout<<"Writing:"<<filename_out<<endl;
    ITKImageWriter->SetFileName(filename_out.c_str());
    ITKImageWriter->SetInput(output);
    ITKImageWriter->Update();
  }
  return EXIT_SUCCESS;

}

int main(int argc, char ** argv)
{

    PARSE_ARGS;

    try
    {
        return execute(inputImageFilename,outputImageFilename,size_vec,resol_vec,flagMask,flagNormalise,flagDoppler);

    }
    catch(std::exception& e)
    {

        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cerr << "Unknown error!" << "\n";
        return EXIT_FAILURE;
    }

}
