#include "NeuroVec.hpp"
#include "Linear.hpp"
#include "LSTM.hpp"
#include "Softmax.hpp"
#include "CrossEntropyLossFunction.hpp"

#include <vector>
#include <utility>
#include <string>
#include <unordered_map>

using namespace std;

vector<string> split(string str, char split)
{
    string temp = "";
    vector<string> res;
    for (int i = 0; i < str.size(); i++)
    {
        if (split == str[i])
        {
            // cout << temp << endl;
            if (temp == "")
                continue;
            res.push_back(temp);
            temp = "";
            continue;
        }
        temp += str[i];
    }
    if (temp != "")
        res.push_back(temp);
    return res;
}

// Extract poem and word vector
pair<vector<vector<string>>, unordered_map<string, int>> MakeData(string pathData)
{
    fstream newFile, wordFile;
    string temp;
    vector<vector<string>> res;
    unordered_map<string, int> wordIndex;

    newFile.open(pathData, ios::in);
    if (!newFile.is_open())
    {
        std::cerr << "Error: Could not open file " << pathData << std::endl;
        exit(0);
    }
    string addString = "";
    int count = 0;

    while (getline(newFile, temp))
    {
        if (temp == "||")
        {
            vector<string> tempSplitedPoem = split(addString, ' ');
            res.push_back(tempSplitedPoem);

            for (int i = 0; i < tempSplitedPoem.size(); i++)
            {
                if (wordIndex.find(tempSplitedPoem[i]) == wordIndex.end())
                {
                    wordIndex[tempSplitedPoem[i]] = count;
                    count += 1;
                }
            }
            addString = "";
            continue;
        }
        if (temp != "")
            addString += (temp + " + ");
    }

    pair<vector<vector<string>>, unordered_map<string, int>> returnData;
    returnData.first = res;
    returnData.second = wordIndex;
    return returnData;
}

class PoemPredict
{
public:
    PoemPredict(vector<vector<string>> &_poemData, unordered_map<string, int> &_wordEmb, int _batchSize = 8, int _hiddenDim = 50) 
    : poemData(_poemData), wordEmb(_wordEmb), batchSize(_batchSize), hiddenDim(_hiddenDim), inLinear(_wordEmb.size(), _hiddenDim), outLinear(_hiddenDim, wordEmb.size()), lstm(hiddenDim)
    {
        inputDim = wordEmb.size();
    }

    vector<NeuroVec<NeuroVec<double>>> CreateSeq(int index)
    {
        int maxDim = 0;
        for (int i = index; i < index + batchSize; i++)
        {
            if (poemData.size() <= i)
                    break;
            if (maxDim < poemData[i].size())
                maxDim = poemData[i].size();
        }

        NeuroVec<NeuroVec<double>> res;
        vector<NeuroVec<NeuroVec<double>>> resSeq;
        for (int i = 0; i < maxDim; i++)
        {
            res = CreateMatrix<double>(batchSize, inputDim, 0);
            for (int j = 0; j < batchSize; j++)
            {
                if (poemData.size() <= index + j)
                    break;
                if (i >= poemData[index + j].size())
                    res[j][wordEmb["+"]] = 1;
                else
                    res[j][wordEmb[poemData[index + j][i]]] = 1;
            }
            resSeq.push_back(res);
        }
        return resSeq;
    }

    vector<NeuroVec<NeuroVec<double>>> CreateTarget(vector<NeuroVec<NeuroVec<double>>> &input)
    {
        vector<NeuroVec<NeuroVec<double>>> res;
        for(int i = 1; i < input.size(); i++)
        {
            res.push_back(CopyMatrix<double>(input[i]));
        }
        return res;
    }

    void Train(int epoch = 10)
    {
        for (int epc = 0; epc < epoch; epc++)
        {
            double totalLoss = 0;
            for (int i = 0; i < poemData.size(); i += batchSize)
            {
                // Forward prapogation
                vector<NeuroVec<NeuroVec<double>>> seqInput = CreateSeq(i);
                vector<NeuroVec<NeuroVec<double>>> seqInRnn, seqOut;
                for(int j = 0; j < seqInput.size(); j++)
                {
                    seqInRnn.push_back(inLinear.Forward(seqInput[j]));
                }

                seqInRnn = lstm.Forward(seqInRnn); // rnn

                for(int j = 0; j < seqInRnn.size(); j++) // output linear and softmax
                {
                    NeuroVec<NeuroVec<double>> temp = outLinear.Forward(seqInRnn[j]);
                    seqOut.push_back(sf.Forward(temp));
                }

                // Target and Loss
                vector<NeuroVec<NeuroVec<double>>> target = CreateTarget(seqInput);
                vector<NeuroVec<double>> crLoss;
                for(int k = 0; k < target.size(); k++)
                {
                    crLoss.push_back(cr.Forward(seqOut[k], target[k]));
                }


                for(int k = 0; k < crLoss.size(); k++)
                {
                    for(int ele = 0; ele < crLoss[k].len; ele++)
                    {
                        totalLoss += crLoss[k][ele];
                    }
                }

                // Baclword prapogation
                vector<NeuroVec<NeuroVec<double>>> prevGrad;
                for(int k = 0; k < target.size(); k++)
                {
                   NeuroVec<NeuroVec<double>> temp = cr.Backward(seqOut[k], target[k]);
                   temp = sf.Backward(temp, seqOut[k]);
                   prevGrad.push_back(outLinear.Backward(temp, seqInRnn[k]));
                }
                prevGrad = lstm.Backward(prevGrad);

                for(int k = 0; k < prevGrad.size(); k++)
                {
                    inLinear.Backward(prevGrad[k], seqInput[k]);
                }
            }
            cout << "Epoch: " << epc + 1 << "| Loss: " << totalLoss << endl;
        }
    }

private:
    vector<vector<string>> poemData;
    unordered_map<string, int> wordEmb;
    int batchSize, hiddenDim, inputDim;
    Linear inLinear, outLinear;
    LSTM lstm;
    Sofmax sf;
    CrossEntropy cr;
};

int main()
{
    pair<vector<vector<string>>, unordered_map<string, int>> data = MakeData("C:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NeuroCpp-LSTM\\glove.6B\\temp.txt");
    PoemPredict model(data.first, data.second);

    cout << "Total Poem: " << data.first.size() << " Total Vocab: " << data.second.size() << endl;
    cout << "Start Training...." << endl;
    model.Train();
    return 0;
}