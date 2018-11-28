#ifndef BestCircleFitT_H
#define BestCircleFitT_H

#include <vnl/algo/vnl_levenberg_marquardt.h>
#include <vnl/vnl_least_squares_function.h>
#include <vector>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

using namespace std;

class BestCircleFit : public vnl_least_squares_function
{
public:    

    BestCircleFit(int unknowns = 1, int nsamples = 1);
    ~BestCircleFit();

    virtual void f(vnl_vector< double > const &x, vnl_vector< double > &fx);

    void SetCenter(vnl_vector<double> center){
        m_Center = center;
    }

    vnl_vector<double> GetCenter(){
        return m_Center;
    }

    void SetRadius(vnl_vector<double> radius){
        m_Radius = radius;
    }

    vnl_vector<double> GetRadius(){
        return m_Radius;
    }

    void SetBoundaryPoints(vector< vnl_vector<double> > boundary_points){
        m_BoundaryPoints = boundary_points;
    }

    vector< vnl_vector<double> > GetBoundaryPoints(){
        return m_BoundaryPoints;
    }

private:

    vector< vnl_vector<double> > m_BoundaryPoints;
    vnl_vector<double> m_BoundaryPoint;
    vnl_matrix< double > m_Transform;
    vnl_vector<double> m_Radius;
    vnl_vector<double> m_Center;
    double m_Angle;
    int m_OptimizationType;
    int m_Samples;

};

#endif // BestEllipseFit_H
