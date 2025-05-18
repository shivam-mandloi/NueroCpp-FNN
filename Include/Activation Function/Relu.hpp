#pragma once

#include <stdexcept>

#include "NeuroVec.hpp"

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

    std::vector<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> prevGrad)
    {
        auto func = [](double x){x > 0 ? 1 : 0;};
    }
private:
    NeuroVec<NeuroVec<double>> savedInput;
};