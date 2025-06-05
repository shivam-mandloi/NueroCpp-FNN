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
        // Print<double>(input);
        // cout << endl;
        input = rl1.Forward(input);
        // Print<double>(input);
        // cout << endl;
        input = li2.Forward(input);
        // Print<double>(input);
        // cout << endl;
        input = rl2.Forward(input);
        // Print<double>(input);
        // cout << endl;
        input = li3.Forward(input);
        // Print<double>(input);
        // cout << endl;
        input = sf.Forward(input);
        // Print<double>(input);
        // cout << endl;
        return input;
    }

    void BackPropagate()
    {
        NeuroVec<NeuroVec<double>> grad = crsLoss.Backward();
        // cout << "cross" << endl;
        // Print<double>(grad);
        // cout << endl;
        // cout << "softmax" << endl;
        grad = sf.Backward(grad);
        // Print<double>(grad);
        // cout << endl;
        // cout << "linear3" << endl;
        grad = li3.Backward(grad);
        // Print<double>(grad);
        // cout << "Relu" << endl;
        grad = rl2.Backward(grad);
        // Print<double>(grad);
        // cout << endl;
        // cout << "linear2" << endl;
        grad = li2.Backward(grad);
        // Print<double>(grad);
        // cout << "Relu" << endl;
        grad = rl1.Backward(grad);
        // Print<double>(grad);
        // cout << endl;
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
        // cout << "Pred: " << endl;
        // Print<double>(input);
        // cout << "Actual" << endl;
        // Print<double>(actual);
        double totalLoss = 0;
        // cout << "Total Loss: " << endl;
        // Print<double>(loss);
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
    
    int batchSize = 16;
    vector<NeuroVec<NeuroVec<double>>> data; 
    vector<NeuroVec<int>> actual;
    NeuroVec<NeuroVec<double>> tempData = CreateMatrix<double>(batchSize, res[0].len - 1, 0);
    NeuroVec<int> tempTarget = CreateVector<int>(batchSize, 0);
    int k = 0;
    for(int i = 0; i < res.size(); i++)
    {
        for(int j = 0; j < res[i].len - 1; j++)
        {
            tempData[k][j] = res[i][j];
        }

        tempTarget[k] = res[i][res[0].len-1];
        k += 1;
        if(k == batchSize)
        {
            data.push_back(tempData);
            actual.push_back(tempTarget);
            tempData = CreateMatrix<double>(batchSize, res[0].len - 1, 0);
            tempTarget = CreateVector<int>(batchSize, 0);
            k = 0;
        }
    }

    nn nueral;
    cout << data.size() << endl;
    for(int epoch = 1; epoch <= 50; epoch++)
    {
        for(int i = 0; i < data.size() - 2; i++)
        {
            NeuroVec<NeuroVec<double>> pred = nueral.Forward(data[i]);            
            double loss = nueral.loss(pred, actual[i]);
            cout << "Epoch: " << epoch << "|" << "loss: " << loss << endl;
            // break;
        }
    }
    cout << "Done" << endl;
    for(int i = data.size() - 2; i < data.size(); i++)
    {
        NeuroVec<NeuroVec<double>> pred = nueral.Forward(data[i]);  
        Print<double>(pred);
        cout << endl;
        Print<int>(actual[i]);
        double loss = nueral.loss(pred, actual[i]);
        cout << endl;
        cout << "Loss: " << loss << endl; 
        // break;
    }
    // NeuroVec<NeuroVec<double>> mat = CreateRandomMatrix<double>(4, 4);
    // Print<double>(mat);
    // NeuroVec<double> vec = CopyVector<double>(mat[0]);
    // vec[0] = -0.1;
    // Print<double>(vec);
    // cout << endl;
    // Print<double>(mat);
    return 0;
}