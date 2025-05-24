#pragma once

#include <stdexcept>

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"

class Relu
{
public:

    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input)
    {
        savedInput = CopyMatrix<double>(input);
        auto func = [](double x) {return x < 0.0 ? 0.0 : x;};
        NeuroVec<NeuroVec<double>> copyInput = CopyMatrix<double>(input);
        ApplyFunction<double>(copyInput, func);
        return copyInput;
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> prevGrad)
    {
        return ReluGradFunction<double>(prevGrad, savedInput);
    }
private:
    NeuroVec<NeuroVec<double>> savedInput;
};