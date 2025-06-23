# pragma once

#include <iostream>
#include <string>

#include "NeuroVec.hpp"


class SGD
{   
public:

    void Update(NeuroVec<NeuroVec<double>> &weight, NeuroVec<double> &bias, NeuroVec<NeuroVec<double>> &weightChange, NeuroVec<double> &biasChange, double lr = 0.001)
    {
        for(int i = 0; i < weight.len; i++)
        {
            for(int j = 0; j < weight[i].len; j++)
            {
                weight[i][j] -= (lr * weightChange[i][j]);
            }
            bias[i] -= (lr * biasChange[i]); // change in bias is prevgrad @ Identity matrix, which make prevGrad to column matrix
        }
    }

};