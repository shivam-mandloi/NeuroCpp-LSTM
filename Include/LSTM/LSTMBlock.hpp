#pragma once
#include <vector>

#include "NeuroVec.hpp"
#include "HadamardProduct.hpp"
#include "Linear.hpp"
#include "Softmax.hpp"
#include "HadamardProduct.hpp"
#include "Tanh.hpp"

class LSTMBlock
{
public:
    void Forward(NeuroVec<NeuroVec<double>> &input, NeuroVec<NeuroVec<double>> &hidden, NeuroVec<NeuroVec<double>> &context)
    {
        NeuroVec<NeuroVec<double>> copyInput, copyHidden;

        savedInputU.push_back(hidden); // S
        savedInputW.push_back(input);  // S

        // Forget Gate
        copyHidden = Uf.Forward(hidden);
        copyInput = Wf.Forward(input);

        NeuroVec<NeuroVec<double>> prob = mat2matAdd<double>(copyInput, copyHidden);
        prob = sf.Forward(prob);
        NeuroVec<NeuroVec<double>> forgetInfo = product.Forward(prob, context); // Forget Info from context

        // Add Gate
        copyHidden = Ug.Forward(hidden);
        copyInput = Wg.Forward(input);

        // Fetch Info
        NeuroVec<NeuroVec<double>> g = mat2matAdd<double>(copyInput, copyHidden);
        g = tanh.Forward(g); // Get Info

        copyHidden = Ui.Forward(hidden);
        copyInput = Wi.Forward(input);

        NeuroVec<NeuroVec<double>> i = mat2matAdd<double>(copyInput, copyHidden);
        i = sf.Forward(i);
        i = product.Forward(g, i); // Fetch data from current input and prev hidden

        // New Context vector
        context = mat2matAdd<double>(forgetInfo, i);

        // Output Gate
        copyHidden = Uo.Forward(hidden);
        copyInput = Wo.Forward(input);
        NeuroVec<NeuroVec<double>> o = mat2matAdd<double>(copyHidden, copyInput);
        o = sf.Forward(o);
        oGateSave.push_back(o);
        savedOtInput1.push_back(o); // S

        hidden = tanh.Forward(context);
        savedOtInput2.push_back(hidden); // S
        hidden = product.Forward(o, hidden);
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &hiddenGrad, NeuroVec<NeuroVec<double>> &contextGrad, int index)
    {
        std::pair<NeuroVec<NeuroVec<double>>, NeuroVec<NeuroVec<double>>> gradOut = product.Backward(hiddenGrad, savedOtInput1[index], savedOtInput2[index]);
        hiddenGrad = sf.Backward(gradOut.first, oGateSave[index]);
    }

private:
    Linear Uf, Wf, Ug, Wg, Ui, Wi, Uo, Wo;
    Sofmax sf;
    std::vector<NeuroVec<NeuroVec<double>>> savedInputU, savedInputW, savedOtInput1, savedOtInput2, oGateSave;
    HadamardProduct product;
    Tanh tanh;
};