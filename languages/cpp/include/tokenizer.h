#pragma once
#include <string>
#include <vector>

// Encode text → byte-level tokens (0–255)
std::vector<int> tokenize(const std::string& s);

// Decode tokens → text
std::string detokenize(const std::vector<int>& tokens);


