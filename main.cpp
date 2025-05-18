#include <vector>
#include "CrossEntropyLossFunction.hpp"

using namespace std;
int main()
{
    CrossEntropy loss;
    NeuroVec<NeuroVec<double>> groundT = CreateMatrix<double>(4, 5, 0);
    groundT[0][3] = 1;
    groundT[1][2] = 1;
    groundT[2][1] = 1;
    groundT[3][4] = 1;

    Print<double>(groundT);
    cout << endl;
    NeuroVec<NeuroVec<double>> predicted = CreateMatrix<double>(4, 5, 0);
    predicted[0][2] = 0.9999;
    predicted[1][1] = 0.9999;
    predicted[2][1] = .9999;
    predicted[3][0] = .9999;
    predicted[0][3] = 0.001;
    predicted[1][2] = 0.001;
    predicted[3][4] = 0.001;
    Print<double>(predicted);

    NeuroVec<double> lossItem = loss.Forward(predicted, groundT);
    Print<double>(lossItem);
    cout << endl;

    NeuroVec<NeuroVec<double>> grad = loss.Backward();
    Print<double>(grad);

    
    return 0;
}