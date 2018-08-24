#ifndef BestEllipseFit_H
#define BestEllipseFit_H

#include <vnl/vnl_least_squares_function.h>
#include <itkImage.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <vector>

using namespace std;

class BestEllipseFit : public vnl_least_squares_function
{
public:    

	typedef unsigned char InputPixelType;
	typedef itk::Image<unsigned char, 3> InputImageType;
	typedef typename InputImageType::IndexType IndexType;
	typedef typename InputImageType::Pointer InputImagePointerType;

	typedef itk::NearestNeighborInterpolateImageFunction<InputImageType> InterpolateImageType;
	typedef typename InterpolateImageType::Pointer InterpolateImagePointerType;

    BestEllipseFit(int nsamples = 100);
    ~BestEllipseFit();

    virtual void f(vnl_vector< double > const &ab, vnl_vector< double > &fx);

    void SetInput(InputImagePointerType img){
    	m_InputImage = img;
    }

    InputImagePointerType GetInput(){
    	return this->m_InputImage;
    }

    // y = sqrt(1 - (x-h)²/a * b²) + k

    void SetTransform(vnl_matrix<double> transform){
    	m_Transform = transform;
    }

    vnl_matrix< double > GetTransform(){
        return m_Transform;
    }

private:

    // vector< vnl_vector<double> > m_Points;
    InputImagePointerType m_InputImage;
    vnl_matrix< double > m_Transform;
    
    int m_Samples;

};

#endif // BestEllipseFit_H
