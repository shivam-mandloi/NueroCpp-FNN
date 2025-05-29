#pragma once

#include <vector>

#include "NeuroVec.hpp"
#include "CrossEntropyLossFunction.hpp"

class Linear
{
public:
    Linear(int inputDim, int outputDim)
    {
        weight = CreateRandomMatrix<double>(outputDim, inputDim);
        bias = CreateRandomVector<double>(outputDim);
    }

    NeuroVec<NeuroVec<double>> Forward(std::vector<NeuroVec<double>> input)
    {
        
    }

    NeuroVec<NeuroVec<double>> Backward(std::vector<NeuroVec<double>> input)
    {

    }
private:
    NeuroVec<NeuroVec<double>> weight;
    NeuroVec<double> bias;
};