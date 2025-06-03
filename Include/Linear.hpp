#pragma once

#include "NeuroVec.hpp"
#include "CrossEntropyLossFunction.hpp"
#include "HelpingFunc.hpp"

class Linear
{
public:
    Linear(int inputDim, int outputDim)
    {
        weight = CreateRandomMatrix<double>(outputDim, inputDim);
        bias = CreateRandomVector<double>(outputDim);
    }

    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input)
    {
        NeuroVec<NeuroVec<double>> output = LinearF(input, weight, bias);
        saveInput = CopyMatrix<double>(input);
        return output;
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad)
    {
        return LinearBAndUpdate(saveInput, prevGrad, weight, bias);
    }
private:
    NeuroVec<NeuroVec<double>> weight;
    NeuroVec<double> bias;
    NeuroVec<NeuroVec<double>> saveInput;
};