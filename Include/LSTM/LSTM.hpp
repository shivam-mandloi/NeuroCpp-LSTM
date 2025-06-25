# pragma once
#include "NeuroVec.hpp"
#include "LSTMBlock.hpp"

#include <vector>


class LSTM
{
public:
    LSTM(int _hiddenDim, int _layer = 1):hiddenDim(_hiddenDim), layer(_layer)
    {
        for(int i = 0; i < layer; i++)
        {
            LSTMBlock block(hiddenDim);
            blocks.push_back(block);
        }

    }
    std::vector<NeuroVec<NeuroVec<double>>> Forward(std::vector<NeuroVec<NeuroVec<double>>> &input)
    {        
        context = CreateMatrix<double>(hiddenDim, hiddenDim, 0);
        hidden = CreateMatrix<double>(hiddenDim, hiddenDim, 0);
        std::vector<NeuroVec<NeuroVec<double>>> res;
        for(int i = 0; i < blocks.size(); i++)
        {
            blocks[i].Forward(input[i], hidden, context);
            res.push_back(hidden);
        }
        return res;
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad)
    {
        
    }
    
private:
    int hiddenDim, layer;
    std::vector<LSTMBlock> blocks;
    NeuroVec<NeuroVec<double>> context, hidden;
};