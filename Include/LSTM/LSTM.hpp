# pragma once
#include "NeuroVec.hpp"
#include "LSTMBlock.hpp"

#include <vector>


class LSTM
{
public:
    LSTM(int _hiddenDim, int _layer = 1):hiddenDim(_hiddenDim), blocks(_hiddenDim){}

    std::vector<NeuroVec<NeuroVec<double>>> Forward(std::vector<NeuroVec<NeuroVec<double>>> &input)
    {   
        NeuroVec<NeuroVec<double>> context = CreateMatrix<double>(input[0].len, hiddenDim, 0); 
        NeuroVec<NeuroVec<double>> hidden = CreateMatrix<double>(input[0].len, hiddenDim, 0);
        std::vector<NeuroVec<NeuroVec<double>>> res;

        for(int i = 0; i < input.size(); i++)
        {
            blocks.Forward(input[i], hidden, context);
            res.push_back(hidden);
        }
        return res;
    }

    std::vector<NeuroVec<NeuroVec<double>>> Backward(std::vector<NeuroVec<NeuroVec<double>>> &prevGrad)
    {        
        NeuroVec<NeuroVec<double>> context = CreateMatrix<double>(prevGrad[0].len, hiddenDim, 0); 
        NeuroVec<NeuroVec<double>> hidden = CreateMatrix<double>(prevGrad[0].len, hiddenDim, 0);

        std::vector<NeuroVec<NeuroVec<double>>> nextGrad;
        for(int i = prevGrad.size()-1; i > -1; i--)
        {
            hidden = mat2matAdd<double>(hidden, prevGrad[i]);
            nextGrad.push_back(blocks.Backward(hidden, context, i));
        }

        blocks.Update();
        return nextGrad;
    }
    
private:
    int hiddenDim;
    LSTMBlock blocks;
};