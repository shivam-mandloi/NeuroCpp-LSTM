#pragma once

#include <random>
#include <vector>
#include <ctime>
#include <functional>
#include <string>
#include <fstream>

#include "NeuroVecCore.hpp"

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

template<typename T>
NeuroVec<NeuroVec<T>> mat2matAdd(NeuroVec<NeuroVec<T>> mat1, NeuroVec<NeuroVec<T>> mat2)
{
    NeuroVec<NeuroVec<T>> res = CreateMatrix<T>(mat1.len, mat1[0].len, 0);
    for(int i = 0; i < mat1.len; i++)
    {
        for(int j = 0; j < mat1[i].len; j++)
        {
            res[i][j] = mat1[i][j] + mat2[i][j];
        }
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

    std::normal_distribution<> gaussian(mean, std::sqrt(variance));
    for(int i = 0; i < vec.len; i++)
    {
        vec[i] = gaussian(gen);
    }
    return scalar2vecMul<double>(0.01, vec);
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
    std::normal_distribution<> gaussian(mean, std::sqrt(variance));

    for(int i = 0; i < mat.len; i++)
    {
        for(int j = 0; j < mat[i].len; j++)
        {
            mat[i][j] = gaussian(gen);
        }
    }
    return mat;
    return scalar2MatMul<double>(0.01, mat);
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
void ApplyFunction(NeuroVec<NeuroVec<T>> &mat, std::function<T(T)> func)
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

template<typename T>
NeuroVec<T> ConvertVectorToNeuroVec(std::vector<T> arr)
{
    NeuroVec<T> resArr = CreateVector<T>(arr.size(), 0);
    for(int i = 0; i < arr.size(); i++)
    {
        resArr[i] = arr[i];
    }
    return resArr;
}

std::vector<double> SplitString(std::string str)
{
    /*
        take string a = "1 2 3 4 5" and return numpy<double> arr = {1, 2, 3, 4, 5} type = double
    */
    std::vector<double> arr;
    std::string temp;
    for(int i = 0; i < str.size(); i++)
    {
        if(str[i] == ' ')
        {
            if(temp != " " && temp != "")
                arr.push_back(std::stod(temp));
            temp = "";
            continue;
        }
        temp += str[i];
    }
    if(temp != "" && temp != " ")
        arr.push_back(std::stod(temp));
    return arr;
}


std::vector<NeuroVec<double>> ReadTxtFile(std::string path)
{
    std::fstream newFile;
    std::string temp;
    std::vector<NeuroVec<double>> res;
    newFile.open(path, std::ios::in);
    if (!newFile.is_open())
    {
        std::cerr << "Error: Could not open file " << path << std::endl;
        exit(0);
    }
    
    while (getline(newFile, temp))
    {
        if (temp != "")
            res.push_back(ConvertVectorToNeuroVec(SplitString(temp)));
    }
    return res;
}

template<typename T>
std::vector<NeuroVec<NeuroVec<T>>> CreateMatrixGroup(std::vector<NeuroVec<T>> &matrix, int group)
{
    std::vector<NeuroVec<NeuroVec<T>>> res;
    NeuroVec<NeuroVec<T>> temp = CreateMatrix<T>(group, matrix[0].len, 0);
    int count = 0;
    for(int i = 0; i < matrix.size(); i++)
    {
        for(int j = 0; j < matrix[i].len; j++)
        {
            temp[count][j] = matrix[i][j];
        }
        count += 1;
        if(count == group)
        {
            count = 0;
            res.push_back(temp);
            temp = CreateMatrix<double>(group, matrix[0].len, 0);
        }
    }
    return res;
}

template<typename T>
std::vector<NeuroVec<T>> CreateVectorGruop(std::vector<NeuroVec<T>> &vec, int group)
{
    std::vector<NeuroVec<T>> res;
    NeuroVec<T> temp = CreateVector<T>(group, 0);
    int count = 0;
    for(int i = 0; i < vec.size(); i++)
    {
        temp[count] = vec[i][0];
        count += 1;
        if(count == group)
        {
            count = 0;
            res.push_back(temp);
            temp = CreateVector<T>(group, 0);
        }
    }
    return res;
}