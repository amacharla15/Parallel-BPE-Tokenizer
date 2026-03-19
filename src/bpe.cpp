#include "bpe.hpp"
#include "byte_encoder.hpp"
#include <climits>
#include <fstream>
#include <iostream>

#include <thread>
#include <algorithm>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include "thread_pool.hpp"

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


std::vector<std::vector<int>> encode_batch(const std::string& path, const TokenizerAssets& assets)
{
    std::ifstream fin(path);

    if (!fin.is_open())
    {
        std::cout << "Could not open file: " << path << std::endl;
        return {};
    }

    std::vector<std::vector<int>> result;
    std::string line;

    while (std::getline(fin, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (line.empty())
        {
            continue;
        }

        std::vector<int> ids = encode_text(line, assets);
        result.push_back(ids);
    }

    return result;
}


std::vector<std::string> read_lines_from_file(const std::string& path)
{
    std::ifstream fin(path);
    if (!fin.is_open())
    {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<std::string> texts;
    std::string line;

    while (std::getline(fin, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        texts.push_back(line);
    }

    return texts;
}

static void encode_range_worker(
    const std::vector<std::string>& texts,
    std::vector<std::vector<int>>& results,
    int start_index,
    int end_index,
    const TokenizerAssets& assets
)
{
    std::cout << "Thread " << std::this_thread::get_id()
              << " started. Range: [" << start_index
              << ", " << end_index << ")" << std::endl;

    for (int i = start_index; i < end_index; i++)
    {
        std::cout << "Thread " << std::this_thread::get_id()
                  << " processing line " << i << std::endl;

        results[i] = encode_text(texts[i], assets);
    }

    std::cout << "Thread " << std::this_thread::get_id()
              << " finished. Range: [" << start_index
              << ", " << end_index << ")" << std::endl;
}

std::vector<std::vector<int>> encode_batch_parallel(
    const std::string& path,
    const TokenizerAssets& assets,
    int num_threads
)
{
    std::vector<std::string> texts = read_lines_from_file(path);

    if (texts.empty())
    {
        return {};
    }

    if (num_threads <= 0)
    {
        num_threads = 1;
    }

    if (num_threads > (int)texts.size())
    {
        num_threads = (int)texts.size();
    }

    std::vector<std::vector<int>> results(texts.size());
    std::vector<std::thread> workers;

    int n = (int)texts.size();
    int chunk_size = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; t++)
    {
        int start_index = t * chunk_size;
        int end_index = std::min(start_index + chunk_size, n);

        if (start_index >= n)
        {
            break;
        }

        workers.emplace_back(
            encode_range_worker,
            std::cref(texts),
            std::ref(results),
            start_index,
            end_index,
            std::cref(assets)
        );
    }

    for (int i = 0; i < (int)workers.size(); i++)
    {
        workers[i].join();
    }

    return results;
}

std::vector<std::vector<int>> encode_batch_thread_pool(
    const std::string& path,
    const TokenizerAssets& assets,
    int num_threads
)
{
    std::vector<std::string> texts = read_lines_from_file(path);

    if (texts.empty())
    {
        return {};
    }

    if (num_threads <= 0)
    {
        num_threads = 1;
    }

    if (num_threads > (int)texts.size())
    {
        num_threads = (int)texts.size();
    }

    std::vector<std::vector<int>> results(texts.size());

    ThreadPool pool((std::size_t)num_threads);

    for (int i = 0; i < (int)texts.size(); i++)
    {
        pool.enqueue([&texts, &results, &assets, i]()
{
            std::cout << "Worker " << std::this_thread::get_id()
                    << " processing line " << i << std::endl;

            results[i] = encode_text(texts[i], assets);

            std::cout << "Worker " << std::this_thread::get_id()
                    << " finished line " << i << std::endl;
        });
    }

    pool.wait_for_all();

    return results;
}