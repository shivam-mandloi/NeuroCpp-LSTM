#include <vector>
#include "NeuroVec.hpp"
#include "Tanh.hpp"
#include "HadamardProduct.hpp"


using namespace std;

int main()
{
    HadamardProduct prod;
    NeuroVec<NeuroVec<double>> vec1 = CreateMatrix<double>(5, 5, 2);
    NeuroVec<NeuroVec<double>> vec2 = CreateMatrix<double>(5, 5, 2);
    NeuroVec<NeuroVec<double>> grad = prod.Forward(vec1, vec2);

    Print<double>(grad);

    cout << endl;
    pair<NeuroVec<NeuroVec<double>>, NeuroVec<NeuroVec<double>>> g = prod.Backward(grad, vec1, vec2);

    Print<double>(g.first);
    cout << endl;
    Print<double>(g.second);
    return 0;
}