#include "volume_generator_gpuCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkAffineTransform.h>
#include <itksys/SystemTools.hxx>
#include <itkTransformFileWriter.h>

#include <itkImageSpatialObject.h>
#include <itkSpatialObjectToImageFilter.h>
#include <itkGroupSpatialObject.h>

#include <uuid/uuid.h>
#include <iostream>
#include <fstream>      // fstream
#include <string>

using namespace std;

typedef unsigned short PixelType;
static const int Dimension = 3;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef InputImageType::IndexType InputImageIndexType;
typedef InputImageType::Pointer InputImagePointerType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::ImageRegionIterator<InputImageType> InputRegionIteratorType;
typedef itk::ImageFileWriter<InputImageType> ImageFileWriterType;

typedef itk::GroupSpatialObject<Dimension> GroupSpatialObjectType;
typedef GroupSpatialObjectType::TransformType TransformType;
typedef itk::ImageSpatialObject<Dimension, PixelType> ImageSpatialObjectType;
typedef itk::SpatialObjectToImageFilter<GroupSpatialObjectType, InputImageType> SpatialObjectToImageFilterType;

typedef map<string, string> CSVLine;

vector<string> splitLine(string line, string delimiter){
	size_t pos = 0;
	string token;
	vector<string> line_split;
	line.erase( std::remove( line.begin(), line.end(), '\r' ), line.end() );
	line.erase( std::remove( line.begin(), line.end(), '\n' ), line.end() );
	while ((pos = line.find(delimiter)) != string::npos) {
	    token = line.substr(0, pos);
	    line_split.push_back(token);
	    line.erase(0, pos + delimiter.length());
	}
	if(line.compare("") != 0){
		line_split.push_back(line);
	}
	return line_split;
}

vector<CSVLine> readCSV(string csvFilename, string csvDelimiter){
	
	vector<CSVLine> csv_parsed;

    ifstream in(csvFilename.c_str());
    if (in.is_open()){
    	string line;
	    map<int, string> csv_pos_header;

	    bool csvHeader = true;
	    int line_count = 0;

	    while(getline(in,line)){
	    	line_count++;
	    	if(csvHeader){
	    		csvHeader = false;
	    		vector<string> line_split = splitLine(line, csvDelimiter);

	    		for(unsigned i = 0; i < line_split.size(); i++){
	    			csv_pos_header[i] = line_split[i];
	    		}
	    	}else{
	    		vector<string> line_split = splitLine(line, csvDelimiter);
	    		CSVLine parsed_line;
	    		for(unsigned i = 0; i < line_split.size(); i++){
	    			if(csv_pos_header.find(i) != csv_pos_header.end()){
	    				parsed_line[csv_pos_header[i]] = line_split[i];
	    			}else{
	    				cerr<<endl<<"Warning! More columns in current line than elements in header, line: "<<line_count<<"column: "<<i+1<<endl;
	    				char buff[50];
	    				sprintf(buff, "%d", i);
	    				parsed_line[buff] = line_split[i];
	    			}
	    		}
	    		csv_parsed.push_back(parsed_line);
	    	}
	    }	
    }else{
    	cerr<<"Cannot read input CSV "<<csvFilename<<endl;
    }

    return csv_parsed;
}

int main (int argc, char * argv[]){


	PARSE_ARGS;

	if(positionDataCsv.compare("") == 0){
		cerr<<"Please provide the csv with the position data"<<endl;
		return EXIT_FAILURE;
	}
    
    
    vector<CSVLine> csv_parsed = readCSV(positionDataCsv, csvDelimiter);

    string image_path = itksys::SystemTools::GetFilenamePath(positionDataCsv);
    vector<string> image_path_components;
    itksys::SystemTools::SplitPath(image_path, image_path_components);

    double conv_angle_factor = M_PI / 180.0;

    InputImageType::PointType min_point;
    min_point[0] = std::numeric_limits<double>::max();
    min_point[1] = std::numeric_limits<double>::max();
    min_point[2] = std::numeric_limits<double>::max();
    InputImageType::PointType max_point;
    max_point[0] = std::numeric_limits<double>::lowest();
    max_point[1] = std::numeric_limits<double>::lowest();
    max_point[2] = std::numeric_limits<double>::lowest();

    char buff[200];

    GroupSpatialObjectType::Pointer groupSO = GroupSpatialObjectType::New();

    for(unsigned i = 0; i < csv_parsed.size(); i++){
    	CSVLine csv_line = csv_parsed[i];

    	double rotX = atof(csv_line[csvColumnRotX].c_str());
    	double rotY = atof(csv_line[csvColumnRotY].c_str());
    	double rotZ = atof(csv_line[csvColumnRotZ].c_str());
    	
    	TransformType::OutputVectorType translateVector;
		translateVector.Fill(0);

    	translateVector[0] = atof(csv_line[csvColumnPosX].c_str());
    	translateVector[1] = atof(csv_line[csvColumnPosY].c_str());
    	translateVector[2] = atof(csv_line[csvColumnPosZ].c_str());

    	if(!angleRadians){
    		rotX *= conv_angle_factor;
    		rotY *= conv_angle_factor;
    		rotZ *= conv_angle_factor;
    	}

    	vector<string> image_filename_components = image_path_components;
    	image_filename_components.push_back(csv_line[csvColumnImageFilename]);
    	string image_filename = itksys::SystemTools::JoinPath(image_filename_components);
    	
    	sprintf(buff, "image: %s, rotX: %f, rotY: %f, rotZ: %f, posX: %f, posY: %f, posZ: %f\n", image_filename.c_str(), rotX, rotY, rotZ, translateVector[0], translateVector[1], translateVector[2]);
    	cout<<buff<<endl;

    	InputImageFileReaderType::Pointer reader = InputImageFileReaderType::New();
    	reader->SetFileName(image_filename);
    	reader->Update();

    	InputImagePointerType img = reader->GetOutput();
    	InputImageType::SpacingType img_spacing_img = img->GetSpacing();
    	if(imgSpacingX > 0){
    		img_spacing_img[0] = imgSpacingX;
    	}
    	if(imgSpacingY > 0){
    		img_spacing_img[1] = imgSpacingY;
    	}
    	if(imgSpacingZ > 0){
    		img_spacing_img[2] = imgSpacingZ;
    	}
    	img->SetSpacing(img_spacing_img);
		
		TransformType::Pointer transform_csv = TransformType::New();
		
		TransformType::OutputVectorType rotVect;
		rotVect.Fill(0);
		rotVect[0] = 1;

		transform_csv->Translate(translateVector);
		transform_csv->Rotate3D(rotVect, rotX);
		rotVect.Fill(0);
		rotVect[1] = 1;
		transform_csv->Rotate3D(rotVect, rotY);

		rotVect.Fill(0);
		rotVect[2] = 1;
		transform_csv->Rotate3D(rotVect, rotZ);


		ImageSpatialObjectType::Pointer imageSO = ImageSpatialObjectType::New();
		imageSO->SetImage(img);
		imageSO->SetObjectToWorldTransform(transform_csv);

		groupSO->AddChild(imageSO);

		imageSO->Update();

		const ImageSpatialObjectType::BoundingBoxType* bb = imageSO->GetMyBoundingBoxInWorldSpace();
		ImageSpatialObjectType::BoundingBoxType::BoundsArrayType bounds = bb->GetBounds();

		min_point[0] = min(min_point[0], min(bounds[0], bounds[1]));
    	min_point[1] = min(min_point[1], min(bounds[2], bounds[3]));
    	min_point[2] = min(min_point[2], min(bounds[4], bounds[5]));

		max_point[0] = max(max_point[0], max(bounds[0], bounds[1]));
    	max_point[1] = max(max_point[1], max(bounds[2], bounds[3]));
    	max_point[2] = max(max_point[2], max(bounds[4], bounds[5]));

    }

    InputImageType::SpacingType spacing;
	spacing[0] = spacingX;
	spacing[1] = spacingY;
	spacing[2] = spacingZ;

	InputImageType::SizeType size;
	size[0] = abs(ceil((max_point[0] - min_point[0])/spacing[0])) + 1;
	size[1] = abs(ceil((max_point[1] - min_point[1])/spacing[1])) + 1;
	size[2] = abs(ceil((max_point[2] - min_point[2])/spacing[2])) + 1;

    SpatialObjectToImageFilterType::Pointer filterSO = SpatialObjectToImageFilterType::New();
    filterSO->SetInput(groupSO);
    filterSO->SetUseObjectValue(true);
	filterSO->SetOutsideValue(0);
	filterSO->SetSpacing(spacing);
	filterSO->SetSize(size);
	filterSO->SetOrigin(min_point);
	filterSO->Update();
	
	InputImagePointerType output_image = filterSO->GetOutput();

	cout<<"Writting image: "<<outFileName<<endl;
	ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
	writer->UseCompressionOn();
	writer->SetFileName(outFileName);
	writer->SetInput(output_image);
	writer->Update();

	return 0;
}