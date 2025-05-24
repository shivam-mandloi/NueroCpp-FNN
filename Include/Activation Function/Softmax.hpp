#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"
#include <cmath>

class Sofmax
{
public:

    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input)
    {
        savedInput = CopyMatrix<double>(input);
        NeuroVec<NeuroVec<double>> copyInput = CopyMatrix<double>(input);
        SoftmaxCalculate(copyInput);
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &input)
    {
        
    }
private:
    NeuroVec<NeuroVec<double>> savedInput;
};