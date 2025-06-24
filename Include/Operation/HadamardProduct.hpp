# pragma once

#include <utility>
#include "NeuroVec.hpp"


class HadamardProduct
{
public:
    // vec1Copy = CreateMatirx<double>(vec1); used as vec1 in Backward parameters
    // vec2Copy = CreateMatirx<double>(vec2); used as vec2 in Backward parameters
    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &vec1, NeuroVec<NeuroVec<double>> &vec2)
    {
        NeuroVec<NeuroVec<double>> res = CreateMatrix<double>(vec1.len, vec1[0].len, 0);
        for(int i = 0; i < vec1.len; i++)
        {
            for(int j = 0; j < vec1.len; j++)
            {
                res[i][j] = vec1[i][j] * vec2[i][j];
            }
        }
        return res;
    }

    std::pair<NeuroVec<NeuroVec<double>>, NeuroVec<NeuroVec<double>>> Backward(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &vec1, NeuroVec<NeuroVec<double>> &vec2)
    {
        std::pair<NeuroVec<NeuroVec<double>>, NeuroVec<NeuroVec<double>>> res;
        NeuroVec<NeuroVec<double>> vec1Grad = CreateMatrix<double>(vec1.len, vec1[0].len, 0);
        NeuroVec<NeuroVec<double>> vec2Grad = CreateMatrix<double>(vec1.len, vec1[0].len, 0);
        for(int i = 0; i < vec1.len; i++)
        {
            for(int j = 0; j < vec2.len; j++)
            {
                vec1Grad[i][j] = prevGrad[i][j] * vec2[i][j];
                vec2Grad[i][j] = prevGrad[i][j] * vec1[i][j];
            }
        }
        res.first = vec1Grad;
        res.second = vec2Grad;
        return res;
    }
private:
};