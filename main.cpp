#include <vector>

#include "CrossEntropyLossFunction.hpp"
#include "MSE.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "SGD.hpp"
#include "Linear.hpp"

using namespace std;

struct nn
{
    Linear li1, li2, li3;
    Relu rl1, rl2, rl3;
    Sofmax sf;
    MSE mseLoss;
    nn():li1(9, 64), li2(64, 32), li3(32, 1){}
    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> input)
    {
        input = li1.Forward(input);
        // cout << "Linear 1" << endl;
        // Print<double>(input);
        input = rl1.Forward(input);
        // cout << "relu 1" << endl;
        // Print<double>(input);
        input = li2.Forward(input);
        // cout << "Linear 2" << endl;
        // Print<double>(input);
        input = rl2.Forward(input);
        // cout << "Relu 2" << endl;
        // Print<double>(input);
        input = li3.Forward(input);
        // cout << "Linear 3" << endl;
        // Print<double>(input);
        // input = rl3.Forward(input);
        // cout << "Relu 3" << endl;
        // Print<double>(input);
        return input;
    }

    void BackPropagate()
    {
        NeuroVec<NeuroVec<double>> grad = mseLoss.Backward();
        // grad = rl3.Backward(grad);
        grad = li3.Backward(grad);
        grad = rl2.Backward(grad);
        grad = li2.Backward(grad);
        grad = rl1.Backward(grad);
        li1.Backward(grad);
    }

    double loss(NeuroVec<NeuroVec<double>> input, NeuroVec<double> actual, bool check = 1)
    {
        NeuroVec<NeuroVec<double>> groundTruth = CreateMatrix<double>(input.len, input[0].len, 0);
        for(int i = 0; i < actual.len; i++)
        {
            groundTruth[i][0] = actual[i];
        }
        NeuroVec<double> loss = mseLoss.Forward(input, groundTruth);
        double totalLoss = 0;
        for(int i = 0; i < loss.len; i++)
        {
            totalLoss += loss[i];
        }       
        if(check)
            BackPropagate();
        return totalLoss/loss.len;
    }
};

int main()
{
    string trainPath = "C:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NueroCpp-FNN\\Housing Data\\trainData.txt";
    string trainTargetPath = "C:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NueroCpp-FNN\\Housing Data\\trainTarget.txt";;
    int batchSize = 32;

    std::vector<NeuroVec<double>> data = ReadTxtFile(trainPath);
    vector<NeuroVec<NeuroVec<double>>> batchTrainingData = CreateMatrixGroup<double>(data, batchSize);

    vector<NeuroVec<double>> target = ReadTxtFile(trainTargetPath);
    vector<NeuroVec<double>> batchTargetData = CreateVectorGruop<double>(target, batchSize);
    
    cout << batchTrainingData.size() << "X" << batchTrainingData[0].len << "X" << batchTrainingData[0][0].len << endl;
    cout << batchTargetData.size() << "X" << batchTargetData[0].len << endl;

    nn neural;
    for(int epoch = 0; epoch < 500; epoch++)
    {
        double totalLoss = 0;
        for(int batch = 0; batch < batchTrainingData.size(); batch++)
        {
            NeuroVec<NeuroVec<double>> pred = neural.Forward(batchTrainingData[batch]);
            double loss = neural.loss(pred, batchTargetData[batch]);

            // cout << "actual" << endl;
            // Print<double>(pred);
            // cout << endl;
            // Print<double>(batchTargetData[batch]);
            // cout <<endl;
            // cout << endl;
            totalLoss += loss;
        }
        cout << "Epoch: " << epoch + 1 << "| Loss: " << totalLoss << endl;
    }
    return 0;
}