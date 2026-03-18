#include "bpe.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

int main()
{
    vector<string> symbols = {"l", "o", "w", "e", "r"};

    unordered_map<string, int> merge_ranks;
    merge_ranks["l o"] = 0;
    merge_ranks["lo w"] = 1;
    merge_ranks["e r"] = 2;
    merge_ranks["low er"] = 3;

    vector<string> result = apply_bpe(symbols, merge_ranks);

    for (int i = 0; i < (int)result.size(); i++)
    {
        cout << result[i] << endl;
    }

    return 0;
}