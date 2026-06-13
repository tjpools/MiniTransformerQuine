#include "tokenizer.h"

std::vector<int> tokenize(const std::string& s) {
    std::vector<int> out;
    out.reserve(s.size());
    for (unsigned char c : s)
        out.push_back(int(c));   // exact byte value 0–255
    return out;
}

std::string detokenize(const std::vector<int>& tokens) {
    std::string s;
    s.reserve(tokens.size());
    for (int t : tokens)
        s.push_back(char(t));    // exact byte reconstruction
    return s;
}
