#include "bpe.hpp"
#include "tokenizer_assets.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main()
{
    string user_input;
    cout << "Enter text: ";
    getline(cin, user_input);

    TokenizerAssets assets;
    load_vocab(assets, "data/vocab.json");
    load_merges(assets, "data/merges.txt");

    vector<string> chunks = simple_split_text(user_input);
    vector<int> ids = encode_text(user_input, assets);

    cout << "Chunks:" << endl;
    for (int i = 0; i < (int)chunks.size(); i++)
    {
        cout << "[" << chunks[i] << "]" << endl;
    }

    cout << "Token IDs:" << endl;
    for (int i = 0; i < (int)ids.size(); i++)
    {
        cout << ids[i] << endl;
    }

    return 0;
}