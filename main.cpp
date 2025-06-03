#include <vector>

#include "CrossEntropyLossFunction.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "SGD.hpp"
#include "Linear.hpp"

using namespace std;

struct nn
{
    Linear li1, li2, li3;
    Relu rl1, rl2;
    Sofmax sf;
    CrossEntropy crsLoss;
    nn():li1(4, 10), li2(10, 10), li3(10, 3){}
    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> input)
    {
        input = li1.Forward(input);
        input = rl1.Forward(input);
        input = li2.Forward(input);
        input = rl2.Forward(input);
        input = li3.Forward(input);
        input = sf.Forward(input);
        return input;
    }

    void BackPropagate()
    {
        NeuroVec<NeuroVec<double>> grad = crsLoss.Backward();
        grad = sf.Backward(grad);
        grad = li3.Backward(grad);
        grad = rl2.Backward(grad);
        grad = li2.Backward(grad);
        grad = rl1.Backward(grad);
        li1.Backward(grad);
    }

    double loss(NeuroVec<NeuroVec<double>> input, NeuroVec<int> index, bool check = 1)
    {
        NeuroVec<NeuroVec<double>> actual = CreateMatrix<double>(input.len, input[0].len, 0);
        for(int i = 0; i < index.len; i++)
        {
            actual[i][index[i]] = 1;
        }
        NeuroVec<double> loss = crsLoss.Forward(input, actual);
        double totalLoss = 0;
        for(int i = 0; i < loss.len; i++) 
        {
            totalLoss+= loss[i];
        }       
        if(check)
            BackPropagate();
        return totalLoss/loss.len;
    }
};

int main()
{
    string path = "C:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NueroCpp-FNN\\Data\\Iris.txt";
    vector<NeuroVec<double>> res = ReadTxtFile(path);
    cout << res.size() << "X" << res[0].len << endl;
    

    int batchSize = 5;
    vector<NeuroVec<NeuroVec<double>>> data; 
    vector<NeuroVec<double>> actual;
    NeuroVec<NeuroVec<double>> tempData = CreateMatrix<double>(batchSize, res[0].len - 1, 0);
    NeuroVec<double> tempTarget = CreateVector<double>(batchSize, 0);
    int k = 0;
    for(int i = 0; i < res.size(); i++)
    {
        cout << "start " << endl;
        Print(tempData);
        for(int j = 0; j < res[i].len - 1; j++)
        {
            tempData[i][j] = res[i][j];
        }
        tempTarget[i] = res[i][res.size()-1];
        k += 1;
        if(k == batchSize)
        {
            data.push_back(CopyMatrix<double>(tempData));
            actual.push_back(tempTarget);
            tempData = CreateMatrix<double>(batchSize, res[0].len - 1, 0);
            tempTarget = CreateVector<double>(batchSize, 0);
            k = 0;
        }
    }

    for(int i = 0; i < data.size(); i++)
    {
        Print(data[i]);
        cout << "-------------" << endl;
        Print(actual[i]);
        cout << "=============" << endl;
    }
    return 0;
}