#include "best_ellipse_fit_t.h"

#include "iostream"

using namespace std;

BestEllipseFitT::BestEllipseFitT(int unknowns, int samples)
    : vnl_least_squares_function(unknowns, samples, no_gradient)
{
    m_Angle = 0;
    m_OptimizationType = 0;
}

BestEllipseFitT::~BestEllipseFitT()
{
    // m_Points.clear();
}

void BestEllipseFitT::fEllipse(vnl_vector< double > const &x, vnl_vector< double > &fx){
    double angle = x[0];

    vnl_vector<double> radius(2);
    radius[0] = x[1];
    radius[1] = x[2];

    vnl_vector<double> center(2);
    center[0] = x[3];
    center[1] = x[4];

    vnl_matrix<double> transform(2,2);
    transform.set_identity();
    transform[0][0] = cos(x[0]);
    transform[1][0] = -sin(x[0]);
    transform[0][1] = sin(x[0]);
    transform[1][1] = cos(x[0]);
    
    vector< vnl_vector<double> > boundary_points = this->GetBoundaryPoints();

    for(int i = 0; i < boundary_points.size(); i++){
        vnl_vector<double> boundary_point(2);
        boundary_point[0] = boundary_points[i][0];
        boundary_point[1] = boundary_points[i][1];

        boundary_point = boundary_point - center;
        boundary_point = boundary_point*transform;

        BestEllipseFitT bestfit = BestEllipseFitT();
        bestfit.SetBoundaryPoint(boundary_point);
        bestfit.SetOptimizationType(1);
        bestfit.SetRadius(radius);

        vnl_levenberg_marquardt levenberg(bestfit);

        vnl_vector<double> best_angle(1);
        best_angle[0] = 0;

        levenberg.minimize(best_angle);
        
        vnl_vector<double> ellipse_point(2);

        ellipse_point[0] = radius[0] * cos(best_angle[0]);
        ellipse_point[1] = radius[1] * sin(best_angle[0]);

        fx[i] = (boundary_point - ellipse_point).magnitude();
    }

    // cout<<"fx:"<<fx<<endl;
}


void BestEllipseFitT::fAngle(vnl_vector< double > const &x, vnl_vector< double > &fx){
    double angle = x[0];

    vnl_vector<double> ellipse_point(2);

    ellipse_point[0] = this->GetRadius()[0] * cos(angle);
    ellipse_point[1] = this->GetRadius()[1] * sin(angle);

    fx[0] = (this->GetBoundaryPoint() - ellipse_point).magnitude();
}

void BestEllipseFitT::f(vnl_vector< double > const &x, vnl_vector< double > &fx){

    if(this->GetOptimizationType() == 0){
        this->fEllipse(x, fx);
    }else{
        this->fAngle(x, fx);
    }
    
}