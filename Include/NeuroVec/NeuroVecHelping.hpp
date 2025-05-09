#pragma once
#include "NeuroVec.hpp"

template <typename T>
NeuroVec<T> mat2vecMul(NeuroVec<T> &mat, NeuroVec<T> &vec)
{
    NeuroVec<T> res(vec.len, 0);
    for (int i = 0; i < mat.dim1; i++)
    {
        T sum = 0;
        for (int j = 0; j < mat.dim2; j++)
        {
            sum += mat.Get(i, j) * vec.Get(j);
        }
        res.Update(sum, i);
    }
    return res;
}

template <typename T>
NeuroVec<T> vec2matMul(NeuroVec<T> &vec, NeuroVec<T> &mat)
{
    NeuroVec<T> res(vec.len, 0);
    for (int i = 0; i < mat.dim2; i++)
    {
        T sum = 0;
        for (int j = 0; j < mat.dim1; j++)
        {
            sum += mat.Get(j, i) * vec.Get(j);
        }
        res.Update(sum, i);
    }
    return res;
}

template <typename T>
T vec2vecMul(NeuroVec<T> &vec1, NeuroVec<T> &vec2)
{
    T res = 0;
    for (int i = 0; i < vec1.dim1; i++)
    {
        res += vec1.Get(i) * vec2.Get(i);
    }
    return res;
}

template <typename T>
NeuroVec<T> scalar2vecMul(T scalar, NeuroVec<T> &vec)
{
    NeuroVec<T> res(vec.len, 0);
    for (int i = 0; i < vec.len; i++)
    {
        res.Update(scalar * vec[i], i);
    }
    return res;
}

template <typename T>
NeuroVec<T> Outer(NeuroVec<T> &vec1, NeuroVec<T> &vec2)
{
    NeuroVec<T> mat = NeuroVec<T>::CreateMatrix(vec1.len, vec2.len, 0);
    for (int i = 0; i < vec1.len; i++)
    {
        for (int j = 0; j < vec2.len; j++)
        {
            mat.Update(vec1.Get(i) * vec2.Get(j), i, j);
        }
    }
    return mat;
}
