#include "volume_generatorCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkEuler3DTransform.h>
#include <itkCompositeTransform.h>
#include <itksys/SystemTools.hxx>
#include <itkTransformFileWriter.h>
#include <itkDivideImageFilter.h>
#include <itkListSample.h>


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
typedef itk::CompositeTransform< double, Dimension > CompositeTransformType;
typedef CompositeTransformType::Pointer CompositeTransformPointerType;
typedef itk::Euler3DTransform<double> TransformType;
typedef TransformType::Pointer TransformPointerType;
typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> DivideImageFilterType;

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

    vector<CompositeTransformPointerType> transform_vector;
    vector<InputImagePointerType> image_vector;

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

    for(unsigned i = 0; i < csv_parsed.size(); i++){
    	CSVLine csv_line = csv_parsed[i];

    	double rotX = atof(csv_line[csvColumnRotX].c_str());
    	double rotY = atof(csv_line[csvColumnRotY].c_str());
    	double rotZ = atof(csv_line[csvColumnRotZ].c_str());
    	// double rotX = 0;
    	// double rotY = 0;
    	// double rotZ = 0;
    	double posX = atof(csv_line[csvColumnPosX].c_str());
    	double posY = atof(csv_line[csvColumnPosY].c_str());
    	double posZ = atof(csv_line[csvColumnPosZ].c_str());

    	if(!angleRadians){
    		rotX *= conv_angle_factor;
    		rotY *= conv_angle_factor;
    		rotZ *= conv_angle_factor;
    	}

    	vector<string> image_filename_components = image_path_components;
    	image_filename_components.push_back(csv_line[csvColumnImageFilename]);
    	string image_filename = itksys::SystemTools::JoinPath(image_filename_components);
    	
    	sprintf(buff, "image: %s, rotX: %f, rotY: %f, rotZ: %f, posX: %f, posY: %f, posZ: %f\n", image_filename.c_str(), rotX, rotY, rotZ, posX, posY, posZ);
    	cout<<buff<<endl;

    	InputImageFileReaderType::Pointer reader = InputImageFileReaderType::New();
    	reader->SetFileName(image_filename);
    	reader->Update();

    	InputImagePointerType img = reader->GetOutput();
    	// double spacing_img[3];
    	// spacing_img[0] = 0.25;
    	// spacing_img[1] = 0.25;
    	// spacing_img[2] = 1;
    	// img->SetSpacing(spacing_img);
    	image_vector.push_back(img);

    	TransformPointerType transform_csv = TransformType::New();
    	transform_csv->SetIdentity();
    	TransformType::OffsetType translate_offset;
    	translate_offset[0] = posX;
    	translate_offset[1] = posY;
    	translate_offset[2] = posZ;
    	transform_csv->Translate(translate_offset);
    	transform_csv->SetRotation(rotX, rotY, rotZ);

    	// InputImageType::RegionType region = img->GetLargestPossibleRegion();
    	// InputImageType::RegionType::SizeType size = region.GetSize();
    	// InputImageType::SpacingType spacing = img->GetSpacing();

    	// InputImageType::PointType origin_point = img->GetOrigin();
    	// InputImageType::PointType end_point;
    	// img->TransformIndexToPhysicalPoint(img->GetLargestPossibleRegion().GetIndex() + img->GetLargestPossibleRegion().GetSize(), end_point);

		// TransformPointerType transform_orig = TransformType::New();
  //   	transform_orig->SetIdentity();
  //   	TransformType::OffsetType translate_offset_origin;
  //   	translate_offset_origin[0] = (end_point[0] - origin_point[0])/2.0;
  //   	translate_offset_origin[1] = end_point[1];
  //   	translate_offset_origin[2] = 0;

    	// cout<<"Origin: "<<origin_point<<endl;
    	// cout<<"Region: "<<img->GetLargestPossibleRegion()<<endl;
    	// cout<<"Translate mid point: "<<translate_offset_origin<<endl;
    	// transform_orig->Translate(translate_offset_origin);

    	CompositeTransformPointerType composite_transform = CompositeTransformType::New();
    	composite_transform->AddTransform(transform_csv);
    	// composite_transform->AddTransform(transform_orig);

    	transform_vector.push_back(composite_transform);

  //   	itk::TransformFileWriterTemplate<double>::Pointer writer = itk::TransformFileWriterTemplate<double>::New();
		// writer->SetInput(composite_transform);
		// writer->SetFileName("transform.tfm");
		// writer->Update();
		
		CompositeTransformType::TransformTypePointer inverse_composite_transform = composite_transform->GetInverseTransform();

		InputRegionIteratorType it(img, img->GetLargestPossibleRegion());
		it.GoToBegin();

		while(!it.IsAtEnd()){
			it.GetIndex();
			InputImageType::PointType point;
			img->TransformIndexToPhysicalPoint(it.GetIndex(), point);

			InputImageType::PointType inverse_point = inverse_composite_transform->TransformPoint(point);

			min_point[0] = min(min_point[0], inverse_point[0]);
	    	min_point[1] = min(min_point[1], inverse_point[1]);
	    	min_point[2] = min(min_point[2], inverse_point[2]);

			max_point[0] = max(max_point[0], inverse_point[0]);
	    	max_point[1] = max(max_point[1], inverse_point[1]);
	    	max_point[2] = max(max_point[2], inverse_point[2]);

	    	++it;
		}
    }

	InputImageType::SpacingType spacing;
	spacing[0] = spacingX;
	spacing[1] = spacingY;
	spacing[2] = spacingZ;

	InputImageType::SizeType size;
	size[0] = abs(ceil((max_point[0] - min_point[0])/spacing[0])) + 1;
	size[1] = abs(ceil((max_point[1] - min_point[1])/spacing[1])) + 1;
	size[2] = abs(ceil((max_point[2] - min_point[2])/spacing[2])) + 1;
	InputImageType::RegionType region;
	region.SetSize(size);
	
	InputImagePointerType output_image = InputImageType::New();
	output_image->SetRegions(region);
	output_image->SetSpacing(spacing);
	output_image->SetOrigin(min_point);
	output_image->Allocate();
	output_image->FillBuffer(0);

	InputImagePointerType output_image_count = InputImageType::New();
	output_image_count->SetRegions(region);
	output_image_count->SetSpacing(spacing);
	output_image_count->SetOrigin(min_point);
	output_image_count->Allocate();
	output_image_count->FillBuffer(1);

	cout<<"Computed image region:"<<endl;
	cout<<"Origin: "<<output_image->GetOrigin()<<endl;
	cout<<"Spacing: "<<output_image->GetSpacing()<<endl;
	cout<<output_image->GetLargestPossibleRegion()<<endl;

	for(unsigned i = 0; i < image_vector.size(); i++){
		CompositeTransformType::TransformTypePointer inverse_composite_transform = transform_vector[i]->GetInverseTransform();

		InputRegionIteratorType it(image_vector[i], image_vector[i]->GetLargestPossibleRegion());
		it.GoToBegin();

		while(!it.IsAtEnd()){
			
			InputImageType::PointType point;
			image_vector[i]->TransformIndexToPhysicalPoint(it.GetIndex(), point);

			InputImageType::PointType inverse_point = inverse_composite_transform->TransformPoint(point);

			InputImageType::IndexType output_index;
			output_image->TransformPhysicalPointToIndex(inverse_point, output_index);
			output_image->SetPixel(output_index, output_image->GetPixel(output_index) + it.Get());

			if(i > 0){
				output_image_count->SetPixel(output_index, output_image_count->GetPixel(output_index) + 1);	
			}

	    	++it;
		}
	}

	cout<<"Avg image: "<<outFileName<<endl;
	DivideImageFilterType::Pointer divide = DivideImageFilterType::New();
	divide->SetInput1(output_image);
	divide->SetInput2(output_image_count);
	divide->Update();
	output_image = divide->GetOutput();

	cout<<"Writting image: "<<outFileName<<endl;
	ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
	writer->SetFileName(outFileName);
	writer->SetInput(output_image);
	writer->Update();

	return 0;
}