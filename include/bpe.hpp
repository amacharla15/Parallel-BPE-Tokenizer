#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tokenizer_assets.hpp"

std::vector<std::string> raw_text_to_symbols(const std::string& text);

std::vector<std::string> get_adjacent_pairs(const std::vector<std::string>& symbols);

std::pair<std::string, int> find_best_pair(
    const std::vector<std::string>& symbols,
    const std::unordered_map<std::string, int>& merge_ranks
);

std::vector<std::string> merge_pair_once(
    const std::vector<std::string>& symbols,
    const std::pair<std::string, int>& best_pair
);

std::vector<std::string> apply_bpe(
    const std::vector<std::string>& symbols,
    const std::unordered_map<std::string, int>& merge_ranks
);

std::vector<int> tokens_to_ids(
    const std::vector<std::string>& token,
    const std::unordered_map<std::string, int>& vocab
);

std::vector<std::string> simple_split_text(const std::string& text);

std::vector<int> encode_chunk(const std::string& chunk, const TokenizerAssets& assets);

std::vector<int> encode_text(const std::string& text, const TokenizerAssets& assets);

std::vector<std::vector<int>> encode_batch(const std::string& path, const TokenizerAssets& assets);


std::vector<std::vector<int>> encode_batch_parallel(
    const std::string& path,
    const TokenizerAssets& assets,
    int num_threads
);

std::vector<std::vector<int>> encode_batch_thread_pool(
    const std::string& path,
    const TokenizerAssets& assets,
    int num_threads
);