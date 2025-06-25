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

        updateWeight = CreateMatrix<double>(outputDim, inputDim, 0);
        updateBias = CreateVector<double>(outputDim, 0);
    }

    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input)
    {
        NeuroVec<NeuroVec<double>> output = LinearF(input, weight, bias);
        // saveInput = CopyMatrix<double>(input);
        return output;
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &saveInput)
    {
        NeuroVec<NeuroVec<double>> dldx;
        std::pair<NeuroVec<NeuroVec<double>>, NeuroVec<double>> res = LinearBAndUpdate(saveInput, prevGrad, weight, bias, dldx);
        updateWeight = mat2matAdd<double>(res.first, updateWeight);
        updateBias = vec2vecAdd<double>(res.second, updateBias);
        return dldx;
    }

    void Update()
    {
        // adm.Update(&weight, &bias, dldw, dldb);
        // sgd.Update(weight, bias, dldw, dldb);
        adam.Update(&weight, &bias, updateWeight, updateBias);
        
        updateWeight = CreateMatrix<double>(weight.len, weight[0].len, 0);
        updateBias = CreateVector<double>(bias.len, 0);
    }
private:
    NeuroVec<NeuroVec<double>> weight, updateWeight;
    NeuroVec<double> bias, updateBias;
    Adam adam;
};