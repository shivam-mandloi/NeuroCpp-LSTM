#pragma once

#include "NeuroVec.hpp"
#include "SGD.hpp"
#include "Adam.hpp"
#include <cmath>

template<typename T>
NeuroVec<T> FindCrossLoss(NeuroVec<NeuroVec<T>> &predicted, NeuroVec<NeuroVec<T>> &groundTruth)
{
    NeuroVec<T> res = CreateVector<T>(predicted.len, 0);
    for(int i = 0; i < predicted.len; i++)
    {
        T temp = 0;
        for(int j = 0; j < predicted[i].len; j++)
        {
            temp += (groundTruth[i][j] * std::log(std::max(predicted[i][j], 1e-15)));
        }
        res[i] = -temp;
    }
    return res;
}

template<typename T>
NeuroVec<NeuroVec<T>> CrossBackProp(NeuroVec<NeuroVec<T>> &predicted, NeuroVec<NeuroVec<T>> &groundTruth)
{
    NeuroVec<NeuroVec<T>> grad = CreateMatrix<T>(predicted.len, predicted[0].len, 0);
    for(int i = 0; i < groundTruth.len; i++)
    {
        for(int j = 0;j < groundTruth[i].len; j++)
        {
            if(groundTruth[i][j])
                grad[i][j] = -groundTruth[i][j] / std::max(predicted[i][j], 1e-30);
        }
    }
    return grad;
}

template<typename T>
NeuroVec<NeuroVec<T>> ReluGradFunction(NeuroVec<NeuroVec<T>> &prevGrad, NeuroVec<NeuroVec<T>> &input)
{
    NeuroVec<NeuroVec<T>> grad = CreateMatrix<T>(prevGrad.len, prevGrad[0].len, 0);
    for(int i = 0; i < prevGrad.len; i++)
    {
        for(int j = 0; j < prevGrad[i].len; j++)
        {
            if(input[i][j] >= 0)
                grad[i][j] = prevGrad[i][j];
        }
    }
    return grad;
}


void SoftmaxCalculate(NeuroVec<NeuroVec<double>> &input)
{
    ClipMatrix<double>(input, -200.0, 200.0);
    auto func = [&](double val)->double{return exp(val);};
    
    ApplyFunction<double>(input, func);
    NeuroVec<double> deno = CreateVector<double>(input.len, 0);
    for(int i = 0; i < input.len; i++)
    {
        double temp = 0;
        for(int j = 0; j < input[i].len; j++)
        {
            temp += input[i][j];
        }
        deno[i] = temp;
    }
    
    for(int i = 0; i < input.len; i++)
    {
        for(int j = 0; j < input[i].len; j++)
        {
            input[i][j] = input[i][j] / deno[i];
        }
    }
}

NeuroVec<NeuroVec<double>> SoftmaxDerivative(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &prob)
{
    NeuroVec<NeuroVec<double>> res = CreateMatrix<double>(prevGrad.len, prob[0].len, 0);
    for(int k = 0; k < prob.len; k++)
    {
        for(int i = 0; i < prob[k].len; i++)
        {
            if(prevGrad[k][i] == 0)
                continue;
            NeuroVec<double> copyProb = CopyVector<double>(prob[k]);
            double probSave = -copyProb[i];
            copyProb[i] = copyProb[i] - 1;
            copyProb = scalar2vecMul<double>(probSave, copyProb);
            for(int j = 0; j < copyProb.len; j++)
            {
                res[k][j] += prevGrad[k][i] * copyProb[j];
            }
        }
    }
    return res;
}

NeuroVec<NeuroVec<double>> LinearF(NeuroVec<NeuroVec<double>> &input, NeuroVec<NeuroVec<double>> &weight, NeuroVec<double> &bias)
{
    NeuroVec<NeuroVec<double>> resMat = CreateMatrix<double>(input.len, weight.len, 0);
    for(int i = 0; i < input.len; i++)
    {
        for(int j = 0; j < weight.len; j++)
        {
            double temp = 0;
            for(int k = 0; k < weight[j].len; k++)
            {
                temp += (weight[j][k] * input[i][k]);
            }
            resMat[i][j] = temp + bias[j];
        }
    }
    return resMat;
}

NeuroVec<NeuroVec<double>> LinearBAndUpdate(NeuroVec<NeuroVec<double>> &input, NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &weight, NeuroVec<double> &bias, Adam adm)
{
    NeuroVec<NeuroVec<double>> dldw = CreateMatrix<double>(weight.len, weight[0].len, 0);
    NeuroVec<double> dldb = CreateVector<double>(bias.len, 0);
    NeuroVec<NeuroVec<double>> dldx = CreateMatrix<double>(input.len, weight[0].len, 0);
    SGD sgd;
    for(int i = 0; i < prevGrad.len; i++)
    {
        for(int j = 0; j < prevGrad[i].len; j++)
        {
            for(int k = 0; k < input[i].len; k++)
            {
                dldw[j][k] += (prevGrad[i][j] * input[i][k]);
            }
        }
    }
    for(int i = 0; i < prevGrad.len; i++)
    {
        for(int j = 0; j < prevGrad[i].len; j++)
        {
            dldb[j] += prevGrad[i][j];
        }
    }
    for(int i = 0; i < prevGrad.len; i++)
    {
        for(int j = 0; j < weight[0].len; j++)
        {
            double temp = 0;
            for(int k = 0; k < prevGrad[i].len; k++)
            {
                temp += (prevGrad[i][k] * weight[k][j]);
            }
            dldx[i][j] = temp;
        }
    }
    adm.Update(&weight, &bias, dldw, dldb);
    // sgd.Update(weight, bias, dldw, dldb);
    return dldx;
}

NeuroVec<double> MseForward(NeuroVec<NeuroVec<double>> &predicted, NeuroVec<NeuroVec<double>> &groundTruth)
{
    NeuroVec<double> res = CreateVector<double>(predicted.len, 0);
    for(int i = 0; i < predicted.len; i++)
    {
        double temp = 0;
        for(int j = 0; j < predicted[i].len; j++)
        {
            temp += pow(groundTruth[i][j] - predicted[i][j], 2);
        }
        res[i] = (1/groundTruth[i].len) * temp;
    }
    // res = scalar2vecMul<double>(1/groundTruth.len, res);
    return res;
}

NeuroVec<NeuroVec<double>> MseBackProp(NeuroVec<NeuroVec<double>> &predicted, NeuroVec<NeuroVec<double>> &groundTruth)
{
    NeuroVec<NeuroVec<double>> res = CreateMatrix<double>(predicted.len, predicted[0].len, 0);
    for(int i = 0; i < predicted.len; i++)
    {
        for(int j = 0; j < predicted[i].len; j++)
        {
            res[i][j] = (-2/groundTruth[i].len) * (groundTruth[i][j] - predicted[i][j]);
        }
    }
    // res = scalar2MatMul<double>(-2/groundTruth.len, res);
    return res;
}

// Tanh function
NeuroVec<NeuroVec<double>> TanhCalculate(NeuroVec<NeuroVec<double>> &input)
{
    ClipMatrix<double>(input, -200.0, 200.0);
    NeuroVec<NeuroVec<double>> res = CreateMatrix<double>(input.len, input[0].len, 0);

    for(int i = 0; i < input.len; i++)
    {
        for(int j = 0; j < input[i].len; j++)
        {
            res[i][j] = (exp(input[i][j]) - exp(-input[i][j])) / (exp(input[i][j]) + exp(-input[i][j]));
        }
    }
    return res;
}


//Tanh Derivative
NeuroVec<NeuroVec<double>> TanhDerCalc(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &lastCalOutput)
{
    NeuroVec<NeuroVec<double>> res = CreateMatrix<double>(lastCalOutput.len, lastCalOutput[0].len, 0);
    for(int i = 0; i < lastCalOutput.len; i++)
    {
        for(int j = 0; j < lastCalOutput[i].len; j++)
        {
            res[i][j] = prevGrad[i][j] * (1 - lastCalOutput[i][j]);
        }
    }
    return res;
}