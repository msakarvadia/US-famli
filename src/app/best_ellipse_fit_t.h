#ifndef BestEllipseFitT_H
#define BestEllipseFitT_H

#include <vnl/algo/vnl_levenberg_marquardt.h>
#include <vnl/vnl_least_squares_function.h>
#include <vector>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

using namespace std;

class BestEllipseFitT : public vnl_least_squares_function
{
public:    

    BestEllipseFitT(int unknowns = 1, int nsamples = 1);
    ~BestEllipseFitT();

    virtual void f(vnl_vector< double > const &x, vnl_vector< double > &fx);

    void fEllipse(vnl_vector< double > const &x, vnl_vector< double > &fx);
    void fAngle(vnl_vector< double > const &x, vnl_vector< double > &fx);

    // y = sqrt(1 - (x-h)²/a * b²) + k
    void SetTransform(vnl_matrix<double> transform){
    	m_Transform = transform;
    }

    vnl_matrix< double > GetTransform(){
        return m_Transform;
    }

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

    void SetAngle(double angle){
        m_Angle = angle;
    }

    double GetAngle(){
        return m_Angle;
    }

    // 0 for angle
    // 1 for radius
    void SetOptimizationType(int type){
        m_OptimizationType = type;
    }

    int GetOptimizationType(){
        return m_OptimizationType;
    }

    void SetBoundaryPoints(vector< vnl_vector<double> > boundary_points){
        m_BoundaryPoints = boundary_points;
    }

    vector< vnl_vector<double> > GetBoundaryPoints(){
        return m_BoundaryPoints;
    }

    void SetBoundaryPoint(vnl_vector<double> boundary_point){
        m_BoundaryPoint = boundary_point;
    }

    vnl_vector<double> GetBoundaryPoint(){
        return m_BoundaryPoint;
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
