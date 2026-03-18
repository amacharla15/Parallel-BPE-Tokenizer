#include "bpe.hpp"
#include <climits>

using namespace std;

vector<string> raw_text_to_symbols(const string& text)
{
    vector<string> symbols;

    for (int i = 0; i < (int)text.size(); i++)
    {
        string one_char = "";
        one_char = one_char + text[i];
        symbols.push_back(one_char);
    }

    return symbols;
}

vector<string> get_adjacent_pairs(const vector<string>& symbols)
{
    int n = symbols.size();
    vector<string> res;

    for (int i = 0; i < n - 1; i++)
    {
        res.push_back(symbols[i] + " " + symbols[i + 1]);
    }

    return res;
}

pair<string, int> find_best_pair(
    const vector<string>& symbols,
    const unordered_map<string, int>& merge_ranks
)
{
    vector<string> pairs = get_adjacent_pairs(symbols);

    int temp_rank = INT_MAX;
    string temp_str = "";

    for (int i = 0; i < (int)pairs.size(); i++)
    {
        if (merge_ranks.find(pairs[i]) != merge_ranks.end())
        {
            if (merge_ranks.at(pairs[i]) < temp_rank)
            {
                temp_str = pairs[i];
                temp_rank = merge_ranks.at(pairs[i]);
            }
        }
    }

    if (temp_str == "")
    {
        return {"", -1};
    }

    return {temp_str, temp_rank};
}

vector<string> merge_pair_once(const vector<string>& symbols, const pair<string, int>& best_pair)
{
    vector<string> ans;
    int n = symbols.size();
    int i = 0;

    while (i < n)
    {
        if (i + 1 < n)
        {
            string current_pair = symbols[i] + " " + symbols[i + 1];

            if (current_pair == best_pair.first)
            {
                ans.push_back(symbols[i] + symbols[i + 1]);
                i = i + 2;
                continue;
            }
        }

        ans.push_back(symbols[i]);
        i = i + 1;
    }

    return ans;
}

vector<string> apply_bpe(
    const vector<string>& symbols,
    const unordered_map<string, int>& merge_ranks
)
{
    vector<string> current_symbols = symbols;

    while (true)
    {
        pair<string, int> best_pair = find_best_pair(current_symbols, merge_ranks);

        if (best_pair.second == -1)
        {
            break;
        }

        current_symbols = merge_pair_once(current_symbols, best_pair);
    }

    return current_symbols;
}

vector<int> tokens_to_ids(
    const vector<string>& token,
    const unordered_map<string, int>& vocab
)
{
    int n = token.size();
    vector<int> res;

    for (int i = 0; i < n; i++)
    {
        if (vocab.find(token[i]) != vocab.end())
        {
            res.push_back(vocab.at(token[i]));
        }
    }

    return res;
}