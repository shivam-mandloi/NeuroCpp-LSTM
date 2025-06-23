# pragma once

#include "NeuroVec.hpp"


class HadamardProduct
{
public:
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

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &savedInput)
    {
        
    }
private:
};