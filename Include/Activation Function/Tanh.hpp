#pragma once

#include <cmath>

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"

class Tanh
{
public:
    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input)
    {
        return TanhCalculate(input);
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &lastCalOutput)
    {   
        return TanhDerCalc(prevGrad, lastCalOutput);
    }
};