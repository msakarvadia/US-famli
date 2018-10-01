#include "best_line_fit.h"

#include "iostream"

using namespace std;

BestLineFit::BestLineFit(int unknowns, int samples)
    : vnl_least_squares_function(unknowns, samples, no_gradient)
{
    m_Samples = samples;
}

BestLineFit::~BestLineFit()
{
    // m_Points.clear();
}



void BestLineFit::f(vnl_vector< double > const &x, vnl_vector< double > &fx){

    vnl_vector< double > p1(x.data_block(), 2);
    vnl_vector< double > p2(x.data_block() + 2, 2);

    for(int i = 0; i < this->GetPoints().size(); i++){
        vnl_vector< double > p0 = this->GetPoints()[i];

        vnl_vector<double> v = p2 - p1;
        vnl_vector<double> s = p0 - p1;

        double angle = acos(dot_product(v, s)/(v.magnitude()*s.magnitude()));
        fx[i] = sin(angle)*s.magnitude();
    }   

}
