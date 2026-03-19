#include "tokenizer_assets.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include<vector>
using namespace std;
using json = nlohmann::json;

void load_vocab(TokenizerAssets& assets, const std::string& path)
{
    ifstream fin(path);

    if (!fin.is_open())
    {
        cout << "Could not open file: " << path << endl;
        return;
    }

    json j;

    try
    {
        fin >> j;
    }
    catch (const json::parse_error& e)
    {
        cout << "JSON parse error in " << path << ": " << e.what() << endl;
        return;
    }

    if (!j.is_object())
    {
        cout << "Invalid vocab JSON: root is not an object" << endl;
        return;
    }

    assets.vocab.clear();

    for (auto it = j.begin(); it != j.end(); ++it)
    {
        if (!it.value().is_number_integer())
        {
            continue;
        }

        string token = it.key();
        int id = it.value().get<int>();
        assets.vocab[token] = id;
    }
}

void load_merges(TokenizerAssets& assets, const std::string& path)
{
    ifstream fin(path);

    if (!fin.is_open())
    {
        cout << "Could not open file: " << path << endl;
        return;
    }

    assets.merge_ranks.clear();

    string line;
    int rank = 0;
    int valid_lines = 0;
    int duplicate_keys = 0;

    while (getline(fin, line))
    {
        if (line.empty())
        {
            continue;
        }

        if (line[0] == '#')
        {
            continue;
        }

        string left = "";
        string right = "";

        int i = 0;
        int n = (int)line.size();

        while (i < n && line[i] != ' ')
        {
            left = left + line[i];
            i++;
        }

        while (i < n && line[i] == ' ')
        {
            i++;
        }

        while (i < n)
        {
            right = right + line[i];
            i++;
        }

        if (left.empty() || right.empty())
        {
            continue;
        }

        valid_lines++;

        string key = left + " " + right;

        if (assets.merge_ranks.find(key) != assets.merge_ranks.end())
        {
            duplicate_keys++;
            cout << "Duplicate key found at rank " << rank
                 << ": " << key << endl;
        }

        assets.merge_ranks[key] = rank;
        rank++;
    }

}

void print_summary(const TokenizerAssets& assets)
{
    cout << "Vocab size: " << assets.vocab.size() << endl;
    cout << "Merge count: " << assets.merge_ranks.size() << endl;

    cout << endl;
    cout << "First 10 vocab entries:" << endl;
    int temp = 0;
    for (const auto& it : assets.vocab)
    {
        cout << it.first << " -> " << it.second << endl;
        temp++;
        if (temp == 10)
        {
            break;
        }
    }

    cout << endl;
    cout << "First 10 merge entries:" << endl;
    temp = 0;
    for (const auto& it : assets.merge_ranks)
    {
        cout << it.first << " -> " << it.second << endl;
        temp++;
        if (temp == 10)
        {
            break;
        }
    }
}   


