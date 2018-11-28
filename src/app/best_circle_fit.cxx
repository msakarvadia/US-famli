#include "best_circle_fit.h"

#include "iostream"

using namespace std;

BestCircleFit::BestCircleFit(int unknowns, int samples)
    : vnl_least_squares_function(unknowns, samples, no_gradient)
{
    
}

BestCircleFit::~BestCircleFit()
{
    // m_Points.clear();
}

void BestCircleFit::f(vnl_vector< double > const &x, vnl_vector< double > &fx){

    double radius_sqr = x[0]*x[0];

    vnl_vector<double> center(2);
    center[0] = x[1];
    center[1] = x[2];
    
    vector< vnl_vector<double> > boundary_points = this->GetBoundaryPoints();

    for(int i = 0; i < boundary_points.size(); i++){
        vnl_vector<double> boundary_point(2);
        boundary_point[0] = boundary_points[i][0];
        boundary_point[1] = boundary_points[i][1];

        boundary_point = boundary_point - center;
        double xh = pow(boundary_point[0], 2);
        double yk = pow(boundary_point[1], 2);

        // (x-h)² + (y-k)² = r²
        fx[i] = (xh + yk - radius_sqr);
    }
}