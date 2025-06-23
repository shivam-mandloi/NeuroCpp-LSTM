#pragma once

#include "NeuroVec.hpp"
#include "CrossEntropyLossFunction.hpp"
#include "HelpingFunc.hpp"
#include "Adam.hpp"

class Linear
{
public:
    Linear(int inputDim, int outputDim) : adam(outputDim, inputDim)
    {
        weight = CreateRandomMatrix<double>(outputDim, inputDim);
        bias = CreateRandomVector<double>(outputDim);
    }

    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input)
    {
        NeuroVec<NeuroVec<double>> output = LinearF(input, weight, bias);
        // saveInput = CopyMatrix<double>(input);
        return output;
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &saveInput)
    {
        return LinearBAndUpdate(saveInput, prevGrad, weight, bias, adam);
    }
private:
    NeuroVec<NeuroVec<double>> weight;
    NeuroVec<double> bias;
    Adam adam;
};