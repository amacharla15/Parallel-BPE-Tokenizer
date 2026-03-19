#pragma once
#include <string>
#include <unordered_map>

struct TokenizerAssets {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, int> merge_ranks;
    std::unordered_map<std::string, int> document;
};


void load_vocab(TokenizerAssets& assets, const std::string& path);
void load_merges(TokenizerAssets& assets, const std::string& path);
void print_summary(const TokenizerAssets& assets);

