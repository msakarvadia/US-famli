#include "best_ellipse_fit.h"

#include "iostream"

using namespace std;

BestEllipseFit::BestEllipseFit(int samples)
    : vnl_least_squares_function(2, samples, no_gradient)
{
    m_Samples = samples;
}

BestEllipseFit::~BestEllipseFit()
{
    // m_Points.clear();
}



void BestEllipseFit::f(vnl_vector< double > const &ab, vnl_vector< double > &fx){

    // (x-h)²/a²+(y-k)²/b²=1
    
    double start_x = -ab[0];
    double end_x = ab[0];

    InterpolateImagePointerType interpolate = InterpolateImageType::New();
    interpolate->SetInputImage(this->GetInput());

    for(int sample = 0; sample < m_Samples; sample++){

        vnl_vector<double> xy(3, 1);

        double ratio = (double)sample/(m_Samples - 1);
        xy[0] = start_x * (1.0 - ratio) + end_x * ratio;

        double xha = xy[0]*xy[0]/(ab[0]*ab[0]);
        xy[1] = sqrt((1.0 - xha)*(ab[1]*ab[1]));
        
        if(!isnan(xy[0]) && !isnan(xy[1])){

            cout<<"xy----:"<<xy<<endl;
            xy = this->GetTransform()*xy;
            
            IndexType xy_index;
            xy_index[0] = round(xy[0]);
            xy_index[1] = round(xy[1]);
            xy_index[2] = 0;

            InputPixelType pix = 0;
            
            if(interpolate->IsInsideBuffer(xy_index)){
                fx[sample] = 1.0 - interpolate->EvaluateAtIndex(xy_index)/255.0;
                cout<<fx[sample]<<endl;
            }else{
                fx[sample] = 1;
            }
        }else{
            fx[sample] = 1;
        }
    }   

}

// double BestEllipseFit::GeodesicDistance(vnl_vector< double > x, vnl_vector< double > y){

//     x.normalize();
//     y.normalize();

//     return acos(dot_product(x, y));

// }
