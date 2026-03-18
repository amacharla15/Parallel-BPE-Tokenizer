#include "bpe.hpp"
#include <iostream>
#include <string>
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

    unordered_map<string, int> vocab;
    vocab["l"] = 1;
    vocab["o"] = 2;
    vocab["w"] = 3;
    vocab["e"] = 4;
    vocab["r"] = 5;
    vocab["lo"] = 6;
    vocab["low"] = 7;
    vocab["er"] = 8;
    vocab["lower"] = 9;

    vector<string> final_tokens = apply_bpe(symbols, merge_ranks);
    vector<int> ids = tokens_to_ids(final_tokens, vocab);

    cout << "Final tokens:" << endl;
    for (int i = 0; i < (int)final_tokens.size(); i++)
    {
        cout << final_tokens[i] << endl;
    }

    cout << "Token IDs:" << endl;
    for (int i = 0; i < (int)ids.size(); i++)
    {
        cout << ids[i] << endl;
    }

    return 0;
}