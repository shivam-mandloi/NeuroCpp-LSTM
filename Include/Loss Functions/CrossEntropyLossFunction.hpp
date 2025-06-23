#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"

class CrossEntropy
{
public:    
    NeuroVec<double> Forward(NeuroVec<NeuroVec<double>> &predicted, NeuroVec<NeuroVec<double>> &groundTruth)
    {
        return FindCrossLoss<double>(predicted, groundTruth);
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevInput, NeuroVec<NeuroVec<double>> &prevGroundTruth)
    {
        return CrossBackProp<double>(prevInput, prevGroundTruth);
    }
};