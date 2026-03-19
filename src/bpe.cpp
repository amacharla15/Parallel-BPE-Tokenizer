#include "bpe.hpp"
#include "byte_encoder.hpp"
#include <climits>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

using namespace std;

vector<string> raw_text_to_symbols(const string& text)
{
    vector<int> bytes = text_to_bytes(text);
    vector<string> symbols = bytes_to_symbols(bytes);
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

    int best_rank = INT_MAX;
    string best_pair = "";

    for (int i = 0; i < (int)pairs.size(); i++)
    {
        auto it = merge_ranks.find(pairs[i]);

        if (it != merge_ranks.end())
        {
            if (it->second < best_rank)
            {
                best_pair = pairs[i];
                best_rank = it->second;
            }
        }
    }

    if (best_pair == "")
    {
        return {"", -1};
    }

    return {best_pair, best_rank};
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
        auto it = vocab.find(token[i]);

        if (it != vocab.end())
        {
            res.push_back(it->second);
        }
    }

    return res;
}

std::vector<std::string> simple_split_text(const std::string& text)
{
    std::vector<std::string> result;

    const char* pattern =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    int errornumber;
    PCRE2_SIZE erroroffset;

    pcre2_code* re = pcre2_compile(
        (PCRE2_SPTR)pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &errornumber,
        &erroroffset,
        NULL
    );

    if (re == NULL)
    {
        return result;
    }

    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);

    PCRE2_SIZE offset = 0;
    PCRE2_SIZE subject_length = (PCRE2_SIZE)text.size();

    while (offset < subject_length)
    {
        int rc = pcre2_match(
            re,
            (PCRE2_SPTR)text.c_str(),
            subject_length,
            offset,
            0,
            match_data,
            NULL
        );

        if (rc < 0)
        {
            break;
        }

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE start = ovector[0];
        PCRE2_SIZE end = ovector[1];

        if (end <= start)
        {
            break;
        }

        result.push_back(text.substr((size_t)start, (size_t)(end - start)));
        offset = end;
    }

    pcre2_match_data_free(match_data);
    pcre2_code_free(re);

    return result;
}
vector<int> encode_chunk(const string& chunk, const TokenizerAssets& assets)
{
    vector<string> symbols = raw_text_to_symbols(chunk);
    vector<string> final_tokens = apply_bpe(symbols, assets.merge_ranks);
    vector<int> ids = tokens_to_ids(final_tokens, assets.vocab);
    return ids;
}

vector<int> encode_text(const string& text, const TokenizerAssets& assets)
{
    vector<string> chunks = simple_split_text(text);
    vector<int> result;

    for (int i = 0; i < (int)chunks.size(); i++)
    {
        vector<int> chunk_ids = encode_chunk(chunks[i], assets);

        for (int j = 0; j < (int)chunk_ids.size(); j++)
        {
            result.push_back(chunk_ids[j]);
        }
    }

    return result;
}