#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

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