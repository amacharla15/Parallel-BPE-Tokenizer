#include "bpe.hpp"
#include "tokenizer_assets.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    TokenizerAssets assets;
    load_vocab(assets, "data/vocab.json");
    load_merges(assets, "data/merges.txt");

    vector<vector<int>> batch_ids = encode_batch("data/document.txt", assets);

    cout << "Batch tokenization results:" << endl;
    for (int i = 0; i < (int)batch_ids.size(); i++)
    {
        cout << "Line " << i + 1 << ":" << endl;
        for (int j = 0; j < (int)batch_ids[i].size(); j++)
        {
            cout << batch_ids[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}