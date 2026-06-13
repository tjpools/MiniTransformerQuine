#pragma once
#include <vector>

struct FeedForward {
    int d_model;
    int d_hidden;

    std::vector<float> W1; // [d_model * d_hidden]
    std::vector<float> b1; // [d_hidden]
    std::vector<float> W2; // [d_hidden * d_model]
    std::vector<float> b2; // [d_model]

    FeedForward(int d_model_, int d_hidden_)
        : d_model(d_model_), d_hidden(d_hidden_),
          W1(d_model_ * d_hidden_),
          b1(d_hidden_),
          W2(d_hidden_ * d_model_),
          b2(d_model_) {}

    // X: [T * d_model]
    // Y: [T * d_model]
    void forward(const std::vector<float>& X, int T, std::vector<float>& Y) const;
};
