#include "bpe.hpp"
#include "tokenizer_assets.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main()
{
    string user_input;
    cout << "Enter one word: ";
    cin >> user_input;

    TokenizerAssets assets;
    load_vocab(assets, "data/vocab.json");
    load_merges(assets, "data/merges.txt");

    vector<string> symbols = raw_text_to_symbols(user_input);
    vector<string> final_tokens = apply_bpe(symbols, assets.merge_ranks);
    vector<int> ids = tokens_to_ids(final_tokens, assets.vocab);

    cout << "Initial symbols:" << endl;
    for (int i = 0; i < (int)symbols.size(); i++)
    {
        cout << symbols[i] << endl;
    }

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