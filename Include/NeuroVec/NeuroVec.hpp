#pragma once

#include <random>
#include <vector>
#include <ctime>
#include <functional>

#include "NeuroVecCore.hpp"

// random function use time as a seed, but if two function call are simultaneously that time not increase.
// for that reason add randCount to time, and increase randCount by one everytime.
static int randCount = 0; 

template <typename T>
NeuroVec<T> mat2vecMul(const NeuroVec<NeuroVec<T>> &mat, const NeuroVec<T> &vec)
{
    NeuroVec<T> res = CreateVector<T>(mat.len, 0.0);
    for(int i = 0; i < mat.len; i++)
    {
        T temp = 0.0;
        for(int j = 0; j < mat[i].len; j++)
        {
            temp += mat[i][j] * vec[i];
        }
        res[i] = temp;
    }
    return res;
}

template <typename T>
NeuroVec<T> vec2matMul(const NeuroVec<T> &vec, const NeuroVec<NeuroVec<T>> &mat)
{
    NeuroVec<T> res = CreateVector<int>(vec.len, 0);
    for (int i = 0; i < mat[0].len; i++)
    {
        T sum = 0;
        for (int j = 0; j < mat.len; j++)
        {
            sum += mat[j][i] * vec[j];
        }
        res[i] = sum;
    }
    return res;
}

template <typename T>
T vec2vecMul(const NeuroVec<T> &vec1, const NeuroVec<T> &vec2)
{
    T res = 0;
    for (int i = 0; i < vec1.len; i++)
    {
        res += vec1[i] * vec2[i];
    }
    return res;
}

template <typename T>
NeuroVec<T> scalar2vecMul(T scalar, const NeuroVec<T> &vec)
{
    NeuroVec<T> res = CreateVector<T>(vec.len, 0);
    for (int i = 0; i < vec.len; i++)
    {
        res[i] = vec[i] * scalar;
    }
    return res;
}

template <typename T>
NeuroVec<NeuroVec<T>> scalar2MatMul(T scalar, const NeuroVec<NeuroVec<T>> &mat)
{
    NeuroVec<NeuroVec<T>> resMat = CreateMatrix<T>(mat.len, mat[0].len, 0);
    for (int i = 0; i < mat.len; i++)
    {
        for(int j = 0; j < mat[0].len; j++)
        {
            resMat[i][j] = scalar * mat[i][j];
        }
    }
    return resMat;
}

template<typename T>
NeuroVec<NeuroVec<T>> Mat2MatMul(NeuroVec<NeuroVec<T>> mat1, NeuroVec<NeuroVec<T>> mat2)
{
    NeuroVec<NeuroVec<T>> resMat = CreateMatrix<T>(mat1.len, mat2[0].len);
    for(int i = 0; i < mat1.len; i++)
    {
        for(int j = 0; j < mat1[i].len; j++)
        {
            T temp = 0;
            for(int k = 0; k < mat2.len; k++)
            {
                temp += mat1[i][k] * mat2[k][j];
            }
            resMat[i][j] = temp;
        }
    }
    return resMat;
}

template<typename T>
NeuroVec<NeuroVec<T>> HadamardOverBatch(NeuroVec<NeuroVec<T>> mat1, NeuroVec<NeuroVec<T>> mat2)
{
    NeuroVec<NeuroVec<T>> res = CreateMatrix<T>(mat1.len, mat1[0].len, 0);
    for(int i = 0; i < mat1.len; i++)
    {
        for(int j = 0; j < mat1[i].len; j++)
        {
            res[i][j]= mat1[i][j] * mat2[i][j];
        }
    }
    return res;
}

template <typename T>
NeuroVec<NeuroVec<T>> Outer(const NeuroVec<T> &vec1, const NeuroVec<T> &vec2)
{
    NeuroVec<NeuroVec<T>> mat = CreateMatrix<T>(vec1.len, vec2.len, 0);
    for (int i = 0; i < vec1.len; i++)
    {
        for (int j = 0; j < vec2.len; j++)
        {
            mat[i][j] = vec1[i] * vec2[j];
        }
    }
    return mat;
}

template<typename T>
NeuroVec<T> CreateRandomVector(int size, double mean = 0, double variance = 1.0)
{
    NeuroVec<T> vec = CreateVector<T>(size, 0);
    std::random_device rd;
    std::time_t currentTime = std::time(nullptr);
    unsigned int uniqueNumber = static_cast<unsigned int>(currentTime) + randCount;
    randCount++;
    std::mt19937 gen(uniqueNumber);

    std::uniform_real_distribution<> gaussian(mean, variance);
    for(int i = 0; i < vec.len; i++)
    {
        vec[i] = gaussian(gen);
    }
    return vec;
}

template<typename T>
NeuroVec<NeuroVec<T>> CreateRandomMatrix(int row, int col, double mean = 0.0, double variance = 1.0)
{
    NeuroVec<NeuroVec<T>> mat = CreateMatrix<T>(row, col, 0.0);
    std::random_device rd;
    std::time_t currentTime = std::time(nullptr);
    unsigned int uniqueNumber = static_cast<unsigned int>(currentTime) + randCount;
    randCount++;
    std::mt19937 gen(uniqueNumber);
    std::uniform_real_distribution<> gaussian(mean, variance);

    for(int i = 0; i < mat.len; i++)
    {
        for(int j = 0; j < mat[i].len; j++)
        {
            mat[i][j] = gaussian(gen);
        }
    }
    return mat;
}

template<typename T>
NeuroVec<NeuroVec<T>> CopyMatrix(const NeuroVec<NeuroVec<T>> &mat)
{
    NeuroVec<NeuroVec<T>> copyMat = CreateMatrix<T>(mat.len, mat[0].len, 0);
    for(int i = 0; i < mat.len; i++)
    {
        for(int j = 0; j < mat[i].len; j++)
        {
            copyMat[i][j] = mat[i][j];
        }
    }
    return copyMat;
}

template<typename T>
NeuroVec<T> CopyVector(const NeuroVec<T> vec)
{
    NeuroVec<T> copyVec = CreateVector<T>(vec.len, 0);
    for(int i = 0; i < vec.len; i++)
    {
        copyVec[i] = vec[i];
    }
    return copyVec;
}

template<typename T>
void ApplyFunction(NeuroVec<NeuroVec<T>> &mat, function<T(T)> func)
{
    for(int i = 0; i < mat.len; i++)
    {
        for(int j = 0; j < mat[i].len; j++)
        {
            mat[i][j] = func(mat[i][j]);
        }
    }
}

template<typename T>
void ClipMatrix(NeuroVec<NeuroVec<T>> &mat, T min, T max)
{
    for (int i = 0; i < mat.len; i++)
    {
        for(int j = 0; j < mat[0].len; j++)
        {
            mat[i][j] = std::min(std::max(min, mat[i][j]), max);
        }
    }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    try
    {
        os << "[";
        for (int i = 0; i < vec.size(); i++)
        {
            os << vec[i];
            if (i != vec.size() - 1)
                os << ", ";
        }
        os << "]";
        os << std::endl;
        return os;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}