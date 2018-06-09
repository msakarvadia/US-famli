
#include "texture_inpaintCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkLabelImageToLabelMapFilter.h>
#include <itkLabelMapMaskImageFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkNeighborhoodIterator.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <itkListSample.h>
#include <itkVariableLengthVector.h>
#include <itkKdTreeGenerator.h>
#include <itkImageRandomConstIteratorWithIndex.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkMedianImageFilter.h>

#include "itkSobelEdgeDetectionImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryBallStructuringElement.h>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_sample.h>

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
    typedef unsigned short InputPixelType;
    static const int dimension = 3;
    typedef itk::Image< InputPixelType, dimension> InputImageType;
    typedef InputImageType::Pointer InputImagePointerType;
    
    typedef itk::Array<double> CNNPixelType;
    typedef itk::Image< CNNPixelType, dimension> CNNImageType;
    typedef CNNImageType::Pointer CNNImagePointerType;

    typedef itk::ImageRegionIterator< InputImageType > ImageRegionIteratorType;
    typedef itk::ImageRandomConstIteratorWithIndex< InputImageType > RandomConstImageRegionIteratorType;

    InputImagePointerType imgin = 0;

    InputImageType::SizeType radius;

    radius[0] = radiusVector[0];
    radius[1] = radiusVector[1];
    radius[2] = radiusVector[2];

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

    typedef itk::ImageDuplicator< InputLabelImageType > InputLabelImageDuplicatorType;
    typedef InputLabelImageDuplicatorType::Pointer InputLabelImageDuplicatorPointerType;

    typedef itk::Image< int, dimension> DistanceImageType;

    typedef itk::ImageFileReader< InputLabelImageType > InputLabelImageFileReaderType;
    typedef InputLabelImageFileReaderType::Pointer InputImageLabelFileReaderPointerType;
    InputImageLabelFileReaderPointerType readerlm = InputLabelImageFileReaderType::New();
    readerlm->SetFileName(inputLabelFilename);
    readerlm->Update();
    InputLabelImagePointerType labelimage = readerlm->GetOutput();

    InputLabelImagePointerType maskImage = 0;
    if(inputMaskFilename.compare("") != 0){
        InputImageLabelFileReaderPointerType readerm = InputLabelImageFileReaderType::New();
        readerm->SetFileName(inputMaskFilename);
        readerm->Update();
        maskImage = readerm->GetOutput();    
    }

    InputImagePointerType outputimg = InputImageType::New();
    outputimg->SetRegions(imgin->GetLargestPossibleRegion());
    outputimg->SetSpacing(imgin->GetSpacing());
    outputimg->SetOrigin(imgin->GetOrigin());
    outputimg->SetDirection(imgin->GetDirection());
    outputimg->Allocate();
    outputimg->FillBuffer(0);


    int vectsize = (radius[0]*2 + 1)*(radius[1]*2 + 1)*(radius[2]*2 + 1);
    //This stuff is to keep count of the neighborhoods hits
    InputImagePointerType countimg = InputImageType::New();
    countimg->SetRegions(imgin->GetLargestPossibleRegion());
    countimg->SetSpacing(imgin->GetSpacing());
    countimg->SetOrigin(imgin->GetOrigin());
    countimg->SetDirection(imgin->GetDirection());
    countimg->Allocate();
    countimg->FillBuffer(0);
    CNNImagePointerType cnnimage = CNNImageType::New();
    cnnimage->SetRegions(imgin->GetLargestPossibleRegion());
    cnnimage->SetSpacing(imgin->GetSpacing());
    cnnimage->SetOrigin(imgin->GetOrigin());
    cnnimage->SetDirection(imgin->GetDirection());
    cnnimage->Allocate();
    CNNPixelType cnndefpixel(vectsize);
    cnndefpixel.Fill(0);
    cnnimage->FillBuffer(cnndefpixel);

    CNNImagePointerType cnndistimage = CNNImageType::New();
    cnndistimage->SetRegions(imgin->GetLargestPossibleRegion());
    cnndistimage->SetSpacing(imgin->GetSpacing());
    cnndistimage->SetOrigin(imgin->GetOrigin());
    cnndistimage->SetDirection(imgin->GetDirection());
    cnndistimage->Allocate();
    CNNPixelType cnndistpixel(vectsize);
    cnndistpixel.Fill(0);
    cnndistimage->FillBuffer(cnndistpixel);

    typedef itk::NeighborhoodIterator<InputImageType> InputImageNeighborhoodIteratorType;
    typedef itk::NeighborhoodIterator<CNNImageType> CNNImageNeighborhoodIteratorType;
    typedef itk::NeighborhoodIterator<InputLabelImageType> InputLabelImageNeighborhoodIteratorType;
    typedef itk::NeighborhoodIterator<DistanceImageType> DistanceImageNeighborhoodIteratorType;
    

    InputImageNeighborhoodIteratorType it(radius, imgin, imgin->GetLargestPossibleRegion());
    InputLabelImageNeighborhoodIteratorType itlb(radius, labelimage, labelimage->GetLargestPossibleRegion());

    InputImageNeighborhoodIteratorType itout(radius, outputimg, outputimg->GetLargestPossibleRegion());
    CNNImageNeighborhoodIteratorType itcnn(radius, cnnimage, cnnimage->GetLargestPossibleRegion());
    CNNImageNeighborhoodIteratorType itcnndist(radius, cnndistimage, cnndistimage->GetLargestPossibleRegion());
    InputImageNeighborhoodIteratorType itcount(radius, countimg, countimg->GetLargestPossibleRegion());

    cout<<"Edge image for weights"<<endl;

    typedef itk::Image<double, dimension>          DoubleImageType;
    
    typedef itk::SobelEdgeDetectionImageFilter <InputImageType, DoubleImageType> SobelEdgeDetectionImageFilterType;
    SobelEdgeDetectionImageFilterType::Pointer sobelFilter = SobelEdgeDetectionImageFilterType::New();
    sobelFilter->SetInput(imgin);

    typedef itk::RescaleIntensityImageFilter< DoubleImageType, DoubleImageType > RescaleFilterType;
    RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(sobelFilter->GetOutput());
    rescaleFilter->SetOutputMinimum(0.0);
    rescaleFilter->SetOutputMaximum(1.0);
    rescaleFilter->Update();

    DoubleImageType::Pointer edge_image = rescaleFilter->GetOutput();

    cout<<"Building kdtree..."<<endl;

    typedef itk::VariableLengthVector< float > MeasurementVectorType;
 
    typedef itk::Statistics::ListSample< MeasurementVectorType > SampleType;
    SampleType::Pointer sample = SampleType::New();
    sample->SetMeasurementVectorSize( vectsize );

    it.GoToBegin();
    itlb.GoToBegin();
    itout.GoToBegin();

    MeasurementVectorType mv(vectsize);
    
    std::vector<double> edge_weights;
    
    while(!it.IsAtEnd() && !itlb.IsAtEnd() && !itout.IsAtEnd()){
        bool mask = true;
        if(maskImage){
            mask = (bool)maskImage->GetPixel(it.GetIndex());
        }
        if(itlb.InBounds() && mask){
            bool addvector = true;
            mv.Fill(0);
            for(int i = 0; i < it.Size() && addvector; i++){

                if(itlb.GetPixel(i) != 0){
                    addvector = false;
                }
                mv[i] = it.GetPixel(i);
            }
            if(addvector){
                sample->PushBack( mv );
                edge_weights.push_back(edge_image->GetPixel(it.GetIndex()));
            }        
        }

        itout.SetCenterPixel(it.GetCenterPixel());
        
        ++itlb;
        ++it;
        ++itout;
    }
    

    typedef itk::Statistics::KdTreeGenerator< SampleType > TreeGeneratorType;
    TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
    treeGenerator->SetSample( sample );
    treeGenerator->SetBucketSize( 16 );
    treeGenerator->Update();

    typedef TreeGeneratorType::KdTreeType TreeType;
    typedef TreeType::NearestNeighbors NeighborsType;
    typedef TreeType::KdTreeNodeType NodeType;

    TreeType::Pointer tree = treeGenerator->GetOutput();

    typedef itk::BinaryBallStructuringElement< InputLabelImageType::PixelType, 3> StructuringElementType;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(radius);
    structuringElement.CreateStructuringElement();

    typedef itk::BinaryDilateImageFilter <InputLabelImageType, InputLabelImageType, StructuringElementType> BinaryDilateImageFilterType;

    BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
    dilateFilter->SetInput(labelimage);
    dilateFilter->SetKernel(structuringElement);
    dilateFilter->SetDilateValue(1);
    dilateFilter->Update();

    InputLabelImagePointerType labelimagedilated = dilateFilter->GetOutput();

    typedef itk::SignedMaurerDistanceMapImageFilter< InputLabelImageType, DistanceImageType > DistanceType;
    DistanceType::Pointer distancefilter = DistanceType::New();
    distancefilter->SetInput(labelimagedilated);
    distancefilter->InsideIsPositiveOn();
    distancefilter->Update();
    DistanceImageType::Pointer distanceimg = distancefilter->GetOutput();

    DistanceImageNeighborhoodIteratorType itdist(radius, distanceimg, distanceimg->GetLargestPossibleRegion());

    typedef itk::StatisticsImageFilter<DistanceImageType> DistanceStatisticsImageFilterType;
    DistanceStatisticsImageFilterType::Pointer distancestats = DistanceStatisticsImageFilterType::New();
    distancestats->SetInput(distanceimg);
    distancestats->Update();
    int maxdistance = distancestats->GetMaximum();

    unsigned int numberOfNeighbors = 1;
    
    cout<<" Querying + build vectors..."<<endl;

    cout<<"Max distance level: "<<maxdistance<<endl;

    typedef itk::ImageFileWriter< InputImageType > InputLabelImageFileWriterType;
    InputLabelImageFileWriterType::Pointer writer = InputLabelImageFileWriterType::New();

    InputLabelImageDuplicatorPointerType duplicator = InputLabelImageDuplicatorType::New();
    duplicator->SetInputImage(labelimage);
    duplicator->Update();
    
    itlb = InputLabelImageNeighborhoodIteratorType(radius, duplicator->GetOutput(), duplicator->GetOutput()->GetLargestPossibleRegion());

    if(distanceStep < 1){
        cerr<<"Distance step cannot be less than 1, you will be stuck forever."<<endl;
        return EXIT_FAILURE;
    }

    cout<<"Distance step: "<<distanceStep<<endl;

    for(int currentdistancelevel = 0; currentdistancelevel <= maxdistance; currentdistancelevel+=distanceStep){

        cout<<endl<<"Distance level: "<<currentdistancelevel<<endl;

        countimg->FillBuffer(0);
        cnnimage->FillBuffer(cnndefpixel);
        cnndistimage->FillBuffer(cnndistpixel);
        

        itlb.GoToBegin();
        itcnn.GoToBegin();
        itcnndist.GoToBegin();
        itcount.GoToBegin();
        itout.GoToBegin();
        itdist.GoToBegin();
        
        while(!itcnn.IsAtEnd() && !itcount.IsAtEnd() && !itlb.IsAtEnd() && !itout.IsAtEnd()){

            mv.Fill(0);

            if(itout.InBounds() && itdist.GetCenterPixel() == currentdistancelevel){
                
                for(int i = 0; i < itout.Size(); i++){
                    mv[i] = itout.GetPixel(i);
                    if(itlb.GetPixel(i) != 0){

                        double mean = itout.GetPixel(i);
                        double stdev = 0;

                        std::vector<double> meanv;

                        InputLabelImageNeighborhoodIteratorType::OffsetType offset = itlb.GetOffset(i);

                        int deltai = offset[0] <= 0? 1:-1;
                        int deltaj = offset[1] <= 0? 1:-1;
                        int deltak = offset[2] <= 0? 1:-1;

                        for(int offi = offset[0]; offi != deltai; offi += deltai){
                            for(int offj = offset[1]; offj != deltaj; offj += deltaj){
                                for(int offk = offset[2]; offk != deltak; offk += deltak){
                                    InputLabelImageNeighborhoodIteratorType::OffsetType off = {offi, offj, offk};
                                    if(itlb.GetPixel(off) == 0){
                                        meanv.push_back(itout.GetPixel(off));
                                    }
                                }
                            }
                        }
                        
                        vnl_vector<double> meanvect(meanv.data(), meanv.size());

                        if(meanvect.size() > 0){
                            mean = meanvect.sum()/meanvect.size();
                        }

                        for(int ii = 0; ii < meanvect.size(); ii++){
                            stdev += pow(meanvect[ii] - mean, 2);
                        }
                        if(meanvect.size() > 0){
                            stdev = sqrt(stdev / meanvect.size());
                        }
                        
                        mv[i] = vnl_sample_normal(mean, stdev);
                    }
                }

                vector<double> distance_v;
                TreeType::InstanceIdentifierVectorType neighbors;
                tree->Search( mv, numberOfNeighbors, neighbors, distance_v);
                MeasurementVectorType mv_n = tree->GetMeasurementVector( neighbors[0] );
                double distance = fabs(distance_v[0]) + 0.0001;

                if(edge_weights.size() < neighbors[0]){
                    distance += weightEdgeFilter*edge_weights[neighbors[0]];
                }

                for(int i = 0; i < itout.Size(); i++){
                    if(itlb.GetPixel(i) != 0){
                        CNNPixelType cnnpix = itcnn.GetPixel(i);
                        cnnpix[itcount.GetPixel(i)] = mv_n[i];
                        itcnn.SetPixel(i, cnnpix);
                        CNNPixelType cnndistpix = itcnndist.GetPixel(i);
                        cnndistpix[itcount.GetPixel(i)] = distance;
                        itcnndist.SetPixel(i, cnndistpix);

                        itcount.SetPixel(i, itcount.GetPixel(i) + 1);
                    }
                }
            }

            ++itlb;
            ++itcnn;
            ++itcnndist;
            ++itcount;
            ++itout;
            ++itdist;
        }

        cout<<"Averaging values..."<<endl;

        itlb.GoToBegin();
        itcnn.GoToBegin();
        itcnndist.GoToBegin();
        itcount.GoToBegin();
        itout.GoToBegin();
        itdist.GoToBegin();

        while(!itcnn.IsAtEnd() && !itcount.IsAtEnd() && !itlb.IsAtEnd() && !itout.IsAtEnd()){
            if(itlb.InBounds() && itlb.GetCenterPixel() != 0 && itcount.GetCenterPixel() > 0){
                
                CNNPixelType cnnpix = itcnn.GetCenterPixel();
                CNNPixelType cnndistpix = itcnndist.GetCenterPixel();
                double wsum = 0.0;
                for(int i = 0; i < itcount.GetCenterPixel(); i++){
                    double w = log(1.0/pow(cnndistpix[i], 2) + 1.0);
                    wsum += w;
                    cnnpix[i] *= w;
                }
                itout.SetCenterPixel((InputPixelType)(cnnpix.sum()/wsum));
                
                if(itdist.GetCenterPixel() <= currentdistancelevel || ((bool) round(vnl_sample_uniform(0.0, 0.6)))){
                    itlb.SetCenterPixel(0);
                }
            }

            ++itlb;
            ++itcnn;
            ++itcnndist;
            ++itcount;
            ++itout;
            ++itdist;
        }

        cout<<"Writing: "<<outputImageFilename<<endl;
        writer->SetFileName(outputImageFilename.c_str());
        writer->SetInput(outputimg);
        writer->Update();
    }

    int max_samples = maxSamples;
        
    cout<<endl<<"Max samples: "<<max_samples<<endl;

    itcnn.GoToBegin();
    itcnndist.GoToBegin();
    itcount.GoToBegin();
    itdist.GoToBegin();
    itout.GoToBegin();

    countimg->FillBuffer(0);
    cnnimage->FillBuffer(cnndefpixel);
    cnndistimage->FillBuffer(cnndistpixel);
    
    cout<<"Setting final values..."<<endl;

    InputImageType::SizeType outsize = outputimg->GetLargestPossibleRegion().GetSize();
    RandomConstImageRegionIteratorType itoutrand = RandomConstImageRegionIteratorType(outputimg, outputimg->GetLargestPossibleRegion());
    itoutrand.SetNumberOfSamples(outsize[0]*outsize[1]*outsize[2]);

    itoutrand.GoToBegin();

    while(!itoutrand.IsAtEnd()){
        
        itcount.SetLocation(itoutrand.GetIndex());
        itdist.SetLocation(itoutrand.GetIndex());

        if(itcount.InBounds() && itcount.GetCenterPixel() <= max_samples && itdist.GetCenterPixel() >= 0){
            
            itout.SetLocation(itcount.GetIndex());
            itcnn.SetLocation(itcount.GetIndex());
            itcnndist.SetLocation(itcount.GetIndex());

            mv.Fill(0);

            for(int i = 0; i < itout.Size(); i++){
                mv[i] = itout.GetPixel(i);
            }

            vector<double> distance_v;
            TreeType::InstanceIdentifierVectorType neighbors;
            tree->Search( mv, numberOfNeighbors, neighbors, distance_v);
            MeasurementVectorType mv_n = tree->GetMeasurementVector( neighbors[0] );
            double distance = fabs(distance_v[0]) + 0.0001;

            if(edge_weights.size() < neighbors[0]){
                distance += weightEdgeFilter*edge_weights[neighbors[0]];
            }

            for(int i = 0; i < itout.Size(); i++){
                if(labelimage->GetPixel(itcount.GetIndex(i)) != 0 && itcount.GetPixel(i) <= max_samples){
                    CNNPixelType cnnpix = itcnn.GetPixel(i);
                    cnnpix[itcount.GetPixel(i)] = mv_n[i];
                    itcnn.SetPixel(i, cnnpix);

                    CNNPixelType cnndistpix = itcnndist.GetPixel(i);
                    cnndistpix[itcount.GetPixel(i)] = distance;
                    itcnndist.SetPixel(i, cnndistpix);

                    itcount.SetPixel(i, itcount.GetPixel(i) + 1);
                }
            }
        }

        ++itoutrand;
    }

    cout<<"Averaging values..."<<endl;
    
    itcnn.GoToBegin();
    itcnndist.GoToBegin();
    itcount.GoToBegin();
    itout.GoToBegin();

    while(!itcnn.IsAtEnd() && !itcount.IsAtEnd() && !itout.IsAtEnd() && !itcnndist.IsAtEnd()){
        if(itcount.GetCenterPixel() > 0){
            
            CNNPixelType cnnpix = itcnn.GetCenterPixel();
            CNNPixelType cnndistpix = itcnndist.GetCenterPixel();
            double wsum = 0.0;
            for(int i = 0; i < itcount.GetCenterPixel(); i++){
                double w = log(1.0/pow(cnndistpix[i], 2) + 1.0);
                wsum += w;
                cnnpix[i] *= w;
            }
            
            itout.SetCenterPixel((InputPixelType)(cnnpix.sum()/wsum));
        }

        
        ++itcnn;
        ++itcnndist;
        ++itcount;
        ++itout;
    }

    cout<<"Median filter..."<<endl;
    typedef itk::MedianImageFilter< InputImageType, InputImageType > MedianFilterType;
    MedianFilterType::InputSizeType medianradius;
    medianradius[0] = radiusMedian[0];
    medianradius[1] = radiusMedian[1];
    medianradius[2] = radiusMedian[2];
    typedef MedianFilterType::Pointer MedianFilterPointerType;
    MedianFilterPointerType median = MedianFilterType::New();
    median->SetInput(outputimg);
    median->SetRadius(medianradius);
    median->Update();
    InputImagePointerType medianout = median->GetOutput();

    itout.GoToBegin();

    while(!itout.IsAtEnd()){
        if(labelimage->GetPixel(itout.GetIndex()) != 0){
            itout.SetCenterPixel(medianout->GetPixel(itout.GetIndex()));
        }
        ++itout;
    }

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(outputimg);
    writer->Update();

    return EXIT_SUCCESS;
}
