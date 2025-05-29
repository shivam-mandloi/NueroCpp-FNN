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

template<typename T>
NeuroVec<NeuroVec<T>> ReluGradFunction(NeuroVec<NeuroVec<T>> &prevGrad, NeuroVec<NeuroVec<T>> &input)
{
    NeuroVec<NeuroVec<T>> grad = CreateMatrix<T>(predicted.len, predicted[0].len, 0);
    for(int i = 0; i < prevGrad.len; i++)
    {
        for(int j = 0; j < prevGrad[i].len; j++)
        {
            if(input[i][j] >= 0)
                grad[i][j] = prevGrad[i][j];
        }
    }
    return grad;
}


void SoftmaxCalculate(NeuroVec<NeuroVec<double>> &input)
{
    NeuroVec<NeuroVec<double>> grad = CreateMatrix<double>(input.len, input[0].len, 0);
    for(int i = 0; i < input.len; i++)
    {
        
        ClipMatrix<double>(input, -200.0, 200.0);
        auto func = [&](double val)->double{return exp(val);};
        
        ApplyFunction<double>(input, func);
        NeuroVec<double> deno = CreateVector<double>(input.len, 0);
        for(int i = 0; i < deno.len; i++)
        {
            double temp = 0;
            for(int j = 0; j < input[i].len; i++)
            {
                temp += input[i][j];
            }
            deno[i] = temp;
        }

        for(int i = 0; i < input.len; i++)
        {
            for(int j = 0; j < input[i].len; i++)
            {
                input[i][j] = input[i][j] / deno[i];
            }
        }
    }
}

NeuroVec<NeuroVec<double>> SoftmaxDerivative(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &prob)
{
    NeuroVec<NeuroVec<double>> res = CreateMatrix<double>(prevGrad.len, prob[0].len, 0);
    for(int k = 0; k < prevGrad.len; k++)
    {
        for(int i = 0; i < prob[k].len; i++)
        {
            NeuroVec<double> copyProb = CopyVector<double>(prob[k]);
            copyProb[i] = copyProb[i] - 1;
            copyProb = scalar2vecMul<double>(-copyProb[i], copyProb);
            res[k][i] = vec2vecMul(copyProb, prevGrad[k]);
        }
    }
    return res;
}

NeuroVec<NeuroVec<double>> Linear(NeuroVec<NeuroVec<double>> &input, NeuroVec<NeuroVec<double>> &weight, NeuroVec<double> &bias)
{
    NeuroVec<NeuroVec<double>> resMat = CreateMatrix<double>(input.len, weight.len, 0);
    for(int i = 0; i < input.len; i++)
    {
        for(int j = 0; j < weight.len; j++)
        {
            double temp = 0;
            for(int k = 0; k < weight[j].len; k++)
            {
                temp += (weight[j][k] * input[i][k]);
            }
            resMat[i][j] = temp + bias[j];
        }
    }
    return resMat;
}