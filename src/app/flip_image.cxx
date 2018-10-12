#include "flip_imageCLP.h"

#include <vtkObjectFactory.h>
#include <vtkActor.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkImageActor.h>
#include <vtkActorCollection.h>
#include <vtkImageMapper3D.h>
#include <vtkMapper.h>

#include <itkImageToVTKImageFilter.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkAffineTransform.h>
#include <itkTransformFileWriter.h>
#include <itkImageFileWriter.h>

//Read Image
typedef unsigned short InputPixelType;
static const int dimension = 2;
typedef itk::Image< InputPixelType, dimension> InputImageType;
typedef InputImageType::Pointer InputImagePointerType;
typedef InputImageType::IndexType IndexType;
typedef itk::ContinuousIndex<double, dimension> ContinuousIndexType;

typedef itk::ImageFileReader< InputImageType > InputImageFileReaderType;
typedef InputImageFileReaderType::Pointer InputImageFileReaderPointerType;

typedef itk::ImageFileWriter< InputImageType > InputLabelImageFileWriterType;

typedef itk::BSplineInterpolateImageFunction<InputImageType, double, double> InterpolateImageType;
typedef typename InterpolateImageType::Pointer InterpolateImagePointerType;

typedef itk::ResampleImageFilter<InputImageType, InputImageType > ResampleImageFilterType;
typedef ResampleImageFilterType::Pointer ResampleImageFilterPointerType;
typedef itk::AffineTransform< double, dimension > TransformType;
typedef TransformType::Pointer TransformPointerType;

typedef itk::TransformFileWriterTemplate< double > TransformWriterType;
typedef TransformWriterType::Pointer TransformWriterPointerType;

typedef itk::ImageToVTKImageFilter<InputImageType>       ImageToVTKImageFilterType;
 

using namespace std;

// Define interaction style
class KeyPressInteractorStyle : public vtkInteractorStyleImage
{
  public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);

    KeyPressInteractorStyle(){
      m_Transform = 0;
      m_ResampleFilter = 0;
    }

    void SetImageResample(ResampleImageFilterPointerType resample_filter){
      m_ResampleFilter = resample_filter;
      m_Transform = const_cast<TransformType*>((TransformType*)m_ResampleFilter->GetTransform());
    }

    void SetToVTKFilter(ImageToVTKImageFilterType::Pointer toVTKFilter){
      m_ToVTKFilter = toVTKFilter;
    }
    
 
    virtual void OnKeyPress() 
    {
      // Get the keypress
      vtkRenderWindowInteractor *rwi = this->Interactor;
      std::string key = rwi->GetKeySym();
 
      // Output the key that was pressed
      std::cout << "Pressed " << key << std::endl;
 
      // Handle an arrow key
      if(key == "Up" || key == "Down"){
        if(m_Transform){
          TransformType::ParametersType params = m_Transform->GetParameters();
          params[3] *= -1;
          m_Transform->SetParameters(params);
          m_ResampleFilter->Update();
          m_ToVTKFilter->Update();
          vtkActorCollection* collection = this->GetCurrentRenderer()->GetActors();
          vtkActor* actor = 0;
          do{
            actor = collection->GetNextActor();
            if(actor){
              actor->GetMapper()->Update();
            }
          }while(actor);
          this->GetCurrentRenderer()->Render();
          this->GetCurrentRenderer()->GetRenderWindow()->Render();
        }
      }

      if(key == "Right" || key == "Left"){
        if(m_Transform){
          TransformType::ParametersType params = m_Transform->GetParameters();
          params[0] *= -1;
          m_Transform->SetParameters(params);
          m_ResampleFilter->Update();
          m_ToVTKFilter->Update();
          vtkActorCollection* collection = this->GetCurrentRenderer()->GetActors();
          vtkActor* actor = 0;
          do{
            actor = collection->GetNextActor();
            if(actor){
              actor->GetMapper()->Update();
            }
          }while(actor);
          this->GetCurrentRenderer()->Render();
          this->GetCurrentRenderer()->GetRenderWindow()->Render();
        }
      }
 
      // Forward events
      vtkInteractorStyleImage::OnKeyPress();
    }

  private:
    TransformPointerType m_Transform;
    ResampleImageFilterPointerType m_ResampleFilter;
    ImageToVTKImageFilterType::Pointer m_ToVTKFilter;
 
};
vtkStandardNewMacro(KeyPressInteractorStyle);
 
int main(int argc, char * argv[])
{

  PARSE_ARGS;

  if(inputImageFilename.compare("") == 0){
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  cout << "The input image is: " << inputImageFilename << endl;

  InputImageFileReaderPointerType reader = InputImageFileReaderType::New();
  reader->SetFileName(inputImageFilename);
  reader->Update();

  InputImagePointerType image = reader->GetOutput();

  TransformPointerType transform = TransformType::New();

  ContinuousIndexType center_index;
  center_index[0] = (image->GetLargestPossibleRegion().GetSize()[0])/2.0;
  center_index[1] = (image->GetLargestPossibleRegion().GetSize()[1])/2.0;

  transform->SetCenter(center_index);

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


  ImageToVTKImageFilterType::Pointer toVTKFilter = ImageToVTKImageFilterType::New();
  toVTKFilter->SetInput(image_resample->GetOutput());
  toVTKFilter->Update();

  // Create an actor
  vtkSmartPointer<vtkImageActor> imageActor = vtkSmartPointer<vtkImageActor>::New();
  imageActor->GetMapper()->SetInputData(toVTKFilter->GetOutput());

  // A renderer and render window
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);

  // An interactor
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);

  vtkSmartPointer<KeyPressInteractorStyle> style = vtkSmartPointer<KeyPressInteractorStyle>::New();
  style->SetImageResample(image_resample);
  style->SetToVTKFilter(toVTKFilter);
  renderWindowInteractor->SetInteractorStyle(style);
  style->SetCurrentRenderer(renderer);

  renderer->AddActor(imageActor);
  renderer->SetBackground(1,1,1); // Background color white

  renderWindow->SetSize(1200,900); //(width, height)
  renderWindow->Render();

  renderWindowInteractor->Start();

  cout<<"Writing: "<<outputFilename<<endl;

  TransformWriterPointerType transform_writer = TransformWriterType::New();
  transform_writer->SetInput(transform);
  transform_writer->SetFileName(outputFilename + ".txt");
  transform_writer->Update();

  return EXIT_SUCCESS;
}