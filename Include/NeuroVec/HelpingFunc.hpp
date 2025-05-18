#pragma once

#include "NeuroVec.hpp"

template<typename T>
NeuroVec<T> FindCrossLoss(NeuroVec<NeuroVec<T>> &predicted, NeuroVec<NeuroVec<T>> &groundTruth)
{
    NeuroVec<T> res = CreateVector<T>(predicted.len, 0);
    for(int i = 0; i < predicted.len; i++)
    {
        T temp = 0;
        for(int j = 0; j < predicted[i].len; j++)
        {
            temp += groundTruth[i][j] * std::log(std::max(predicted[i][j], 1e-15));
        }
        res[i] = -temp;
    }
    return res;
}

template<typename T>
NeuroVec<NeuroVec<T>> CrossBackProp(NeuroVec<NeuroVec<T>> &predicted, NeuroVec<NeuroVec<T>> &groundTruth)
{
    NeuroVec<NeuroVec<T>> grad = CreateMatrix<T>(predicted.len, predicted[0].len, 0);
    for(int i = 0; i < groundTruth.len; i++)
    {
        for(int j = 0;j < groundTruth[i].len; j++)
        {
            if(groundTruth[i][j])
                grad[i][j] = -groundTruth[i][j] / std::max(predicted[i][j], 1e-15);
        }
    }
    return grad;
}

