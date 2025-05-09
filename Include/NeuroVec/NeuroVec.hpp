#pragma once

// #include "NeuroVecHelping.hpp"
#include <vector>
#include <iostream>

/*
    => Use array to store vector and matrix.
    => There are two initialization method first by vector and second by size and default value.
    => Use template to create any data type array, Not for complex type.
    => To use more complex array can use vector.
    => Take dataype for only scalar values, like int, float, double, long int etc.
    => Only able to store vector and matrix, not high dim tensor.
    => Use 1D array to store both vector and matrix, means matrix also store in 1D array.
    => while fetching from matrix, dim1 and dim2 variable used as a size in x and y.
    => formula to fetch and update  => array[(i * dim2) + j], when NueroVec is matrix.    
    => Use Get and Update function to fetch and update data in array.
    => To differentiate between matrix and vec use bool variable.
    => For transpose use bool variable, when fetch and update change (i, j) to (j, i).
*/

/*
Usage:

    vector<int> data = {4,2,3};
    NeuroVec<int> test(data);
    test.Print(); // 4 2 3

    cout << test.dim() << endl << endl; // [3]

    vector<vector<int>> vec1 = {{1,2,3},{4,5,6}, {1,1,1}};
    NeuroVec<int> test1(vec1, 3, 3);
    test1.Print(); //1 2 3
                   //4 5 6
                   //1 1 1

    cout << test1.dim() << endl << endl; //[3, 3]

    test1.Trans();
    test1.Print(); //1 4 1
                   //2 5 1
                   //3 6 1
    cout << test1.dim() << endl << endl; //[3, 3]

    mat2vecMul<int>(test1, test).Print(); // 15 21 27
    cout << endl;
    vec2matMul<int>(test, test1).Print(); // 17 44 9
    scalar2vecMul<int>(2.0, test).Print(); // 8 4 6

    test.Print(); //4 2 3
    test.Print(); // 4 2 3
    cout << endl;
    Outer<int>(test, test).Print(); // 16 8 12
                                    // 8 4 6
                                    // 12 6 9

    // Create marix with size (2, 3) with intial value is 0
    NeuroVec<T> mat = NeuroVec<T>::CreateMatrix(2, 3, 0);
*/

template <typename T>
class NeuroVec
{
public:
    bool isMatrix = false; // To check if NueroVec for matrix
    int len = 0; // in case of matrix dim1 * dim2
    int dim1 = 0, dim2 = 0;

    // free array
    ~NeuroVec()
    {
        delete array;
    }

    // Initialization vec
    NeuroVec(int size, T data)
    {
        dim1 = size;
        array = new T[size];
        len = size;
        for (int i = 0; i < len; i++)
        {
            array[i] = data;
        }
    }

    // Initialization vec using std::vector
    NeuroVec(std::vector<T> &vec) : NeuroVec(vec.size(), 0.0)
    {
        for (int i = 0; i < len; i++)
        {
            array[i] = vec[i];
        }
    }

    // Initialization matrix by std::vector
    NeuroVec(std::vector<std::vector<T>> &mat, int x, int y)
    {
        dim1 = x; dim2 = y;
        len = x * y;
        array = new T[len];
        for(int i = 0; i < x; i ++)
        {
            for(int j = 0; j < y; j++)
            {
                array[(i * dim2) + j] = mat[i][j];
            }
        }
        isMatrix = true;
    }

    // static function use to initialization matrix
    static NeuroVec<T> CreateMatrix(int index1, int index2, T data)
    {
        std::vector<std::vector<T>> mat(index1, std::vector<T>(index2, data));
        return NeuroVec<T>(mat, index1, index2);
    }

    // Only work for vec, will change to update
    const T operator[](int index) const
    {
        return array[index];
    }

    // fetch matrix element at (i, j)
    T Get(int index1, int index2)
    {
        if(isMatrix) // check if store array is matrix or vector
        {
            if(isTranspose) // if transpose change  requested index (i, j) to (j, i)
            {
                int temp = index1;
                index1 = index2;
                index2 = temp;
            }
            return array[(index1 * dim2) + index2]; // matrix size is (dim1 X dim2)
        }
        return NULL; // Give error not able to change Null to int, so remove it
    }

    // Fetch element, work for only vector
    T Get(int index)
    {
        return array[index];
    }

    // return dim in std::vector, if transpose return transposed matrix dim
    // same function used to get vector dim to
    std::vector<int> dim()
    {
        std::vector<int> retrnDim;
        if(isMatrix)
        {
            if(isTranspose)
            {
                retrnDim.push_back(dim2);
                retrnDim.push_back(dim1);
            }
            else
            {
                retrnDim.push_back(dim1);
                retrnDim.push_back(dim2);
            }
        }
        else
            retrnDim.push_back(dim1);
        return retrnDim;
    }

    void Trans()
    {
        isTranspose = true;
    }

    // Work for both vector and matrix
    void Print()
    {
        if(isMatrix)
        {
            int x = dim1, y = dim2;
            if(isTranspose)
            {
                x = dim2; y = dim1;
            }
            
            for(int i = 0; i < x; i++)
            {
                for(int j = 0; j < y; j++)
                {
                    std::cout << Get(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            for(int i = 0; i < len; i++)
            {
                std::cout << array[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    // update the element at index i
    // work only for vector
    void Update(T data, int index)
    {
        array[index] = data;
    }

    // work for matrix
    // update value, if matrix is transposed it keep this into consider and update
    void Update(T data, int index1, int index2)
    {
        if(isMatrix)
        {
            if(isTranspose)
            {
                int temp = index1;
                index1 = index2;
                index2 = temp;
            }
            array[(index1 * dim2) + index2] = data;
        }        
    }

private:
    T *array;
    bool isTranspose = false;
};

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
        return os;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}