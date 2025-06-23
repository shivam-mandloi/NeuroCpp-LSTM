#pragma once

#include <iostream>


// NueroVec basic structure
template <typename T>
struct NeuroVec
{
    T *array = nullptr;
    int len = 0;

    ~NeuroVec()
    {
        delete[] array;
    }
    NeuroVec(){}
    
    NeuroVec(int _len): len(_len)
    {
        array = new T[_len];
    }

    NeuroVec(int _len, T defaultVal): len(_len)
    {
        array = new T[_len];
        std::fill(array, array+len, defaultVal);
    }

    NeuroVec(const NeuroVec<T>&otherVec)
    {
        len = otherVec.len;
        array = new T[len];
        std::copy(otherVec.array, otherVec.array+len, array);
    }

    const T &operator[](int index) const
    {
        return array[index];
    }

    T &operator[](int index)
    {
        return array[index];
    }

    NeuroVec<T> &operator=(const NeuroVec<T> &other)
    {
        if (this != &other)
        {
            delete[] array;
            len = other.len;    
            array = new T[len];
            std::copy(other.array, other.array + len, array);
        }
        return *this;
    }
};

template <typename T>
void Print(NeuroVec<T> &vec)
{
    for (int i = 0; i < vec.len; i++)
    {
        std::cout << vec.array[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void Print(NeuroVec<NeuroVec<T>> mat)
{
    for (int i = 0; i < mat.len; i++)
    {
        for (int j = 0; j < mat[i].len; j++)
        {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
NeuroVec<T> CreateVector(size_t N, T defaultData)
{
    NeuroVec<T> res(N, defaultData);
    return res;
}

template <typename T>
NeuroVec<NeuroVec<T>> CreateMatrix(size_t N, size_t M, T defaultData)
{
    NeuroVec<NeuroVec<T>> res(N);
    for(int i = 0; i < N; i++)
    {
        res[i] = CreateVector<T>(M, defaultData);
    }
    return res;
}