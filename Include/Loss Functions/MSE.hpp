#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"

class MSE
{
public:
    NeuroVec<double> Forward(NeuroVec<NeuroVec<double>> &predicted, NeuroVec<NeuroVec<double>> &groundTruth)
    {
        return MseForward(predicted, groundTruth);
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevInput, NeuroVec<NeuroVec<double>> &prevGroundTruth)
    {
        return MseBackProp(prevInput, prevGroundTruth);
    }
};