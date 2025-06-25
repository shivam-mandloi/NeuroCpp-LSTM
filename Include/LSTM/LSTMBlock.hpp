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
    int hiddenDim;
    LSTMBlock(int hidden): Uf(hidden, hidden), Wf(hidden, hidden), Ug(hidden, hidden), Wg(hidden, hidden), Ui(hidden, hidden), Wi(hidden, hidden), Uo(hidden, hidden), Wo(hidden, hidden), hiddenDim(hidden){}

    // copy LSTMBlock
    LSTMBlock(const LSTMBlock &other):
    Uf(other.hiddenDim, other.hiddenDim), Wf(other.hiddenDim, other.hiddenDim), Ug(other.hiddenDim, other.hiddenDim), Wg(other.hiddenDim, other.hiddenDim), Ui(other.hiddenDim, other.hiddenDim), Wi(other.hiddenDim, other.hiddenDim), Uo(other.hiddenDim, other.hiddenDim), Wo(other.hiddenDim, other.hiddenDim), hiddenDim(other.hiddenDim)
    {}

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
        forgetPrI1.push_back(prob);
        forgetPrI2.push_back(context);
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
        prdctGtI1.push_back(g); // s
        prdctGtI2.push_back(i); // s
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
        // consist all the grad of loss wrt to hidden, context and input
        NeuroVec<NeuroVec<double>> nextHiddenGrad, nextINputGrad;

        // Output Gate
        std::pair<NeuroVec<NeuroVec<double>>, NeuroVec<NeuroVec<double>>> gradOut = product.Backward(hiddenGrad, savedOtInput1[index], savedOtInput2[index]);
        hiddenGrad = sf.Backward(gradOut.first, oGateSave[index]);

        nextHiddenGrad = Uo.Backward(hiddenGrad, savedInputU[index]);
        nextINputGrad = Wo.Backward(hiddenGrad, savedInputW[index]);

        NeuroVec<NeuroVec<double>> dldc = tanh.Backward(gradOut.second, savedOtInput2[index]);
        dldc = mat2matAdd<double>(contextGrad, dldc);

        // Add Gate
        gradOut = product.Backward(dldc, prdctGtI1[index], prdctGtI2[index]);
        hiddenGrad = sf.Backward(gradOut.second, prdctGtI2[index]);

        nextHiddenGrad = mat2matAdd<double>(nextHiddenGrad, Ui.Backward(hiddenGrad, savedInputU[index]));
        nextINputGrad = mat2matAdd<double>(nextINputGrad, Wi.Backward(hiddenGrad, savedInputW[index]));
        
        NeuroVec<NeuroVec<double>> dldi = tanh.Backward(gradOut.first, prdctGtI1[index]);

        nextHiddenGrad = mat2matAdd<double>(nextHiddenGrad, Ui.Backward(dldi, savedInputU[index]));
        nextINputGrad = mat2matAdd<double>(nextINputGrad, Wi.Backward(dldi, savedInputW[index]));
        
        // Forget Gate
        gradOut = product.Backward(dldc, forgetPrI1[index], forgetPrI2[index]);
        hiddenGrad = sf.Backward(gradOut.first, forgetPrI1[index]);

        nextHiddenGrad = mat2matAdd<double>(nextHiddenGrad, Uf.Backward(hiddenGrad, savedInputU[index]));
        nextINputGrad = mat2matAdd<double>(nextINputGrad, Wf.Backward(hiddenGrad, savedInputW[index]));

        hiddenGrad = nextHiddenGrad;
        contextGrad = gradOut.second;
        return nextINputGrad;
    }

    void Update()
    {
        Uf.Update(); Wf.Update(); Ug.Update(); Wg.Update(); Ui.Update(); Wi.Update(); Uo.Update(); Wo.Update();
    }

private:
    Linear Uf, Wf, Ug, Wg, Ui, Wi, Uo, Wo;
    Sofmax sf;
    std::vector<NeuroVec<NeuroVec<double>>> savedInputU, savedInputW, savedOtInput1, savedOtInput2, oGateSave, prdctGtI1, prdctGtI2, forgetPrI1, forgetPrI2;
    HadamardProduct product;
    Tanh tanh;
};