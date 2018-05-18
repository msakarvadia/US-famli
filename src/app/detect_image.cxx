
#include "detect_imageCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkRGBToLuminanceImageFilter.h>

#include <array>
using namespace std;

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

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

    typedef itk::LabelStatisticsImageFilter< InputImageType, InputLabelImageType > LabelStatisticsImageFilterType;
    typedef LabelStatisticsImageFilterType::Pointer LabelStatisticsImageFilterPointerType;

    LabelStatisticsImageFilterPointerType labelstats = LabelStatisticsImageFilterType::New();


    labelstats->SetInput(imgin);
    labelstats->SetLabelInput(labelimage);

    labelstats->Update();

    ostringstream outstats;

    outstats << "{";
    outstats << "\"filename\":\""<< inputImageFilename <<"\",";
    outstats << "\"sum\":"<< labelstats->GetSum( labelValue );
    outstats << " }";
    outstats << std::endl;


    if(outputFilename.compare("") != 0){
        ofstream outputstatfile;
        outputstatfile.open(outputFilename.c_str());
        if(outputstatfile.is_open()){
            cout<<"Writing: "<<outputFilename<<endl;
            outputstatfile<< outstats.str()<<endl;
        }else{
            cout<<"Could not create file: "<<outputFilename<<endl;
            return EXIT_FAILURE;
        }
    }else{
        cout << outstats.str();
    }


    return EXIT_SUCCESS;
}
