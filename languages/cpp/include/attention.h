#pragma once
#include <vector>

struct AttentionHead {
    int d_model;
    int d_head;
    int max_seq;

    // Parameters: [d_model * d_head]
    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;

    AttentionHead(int d_model_, int d_head_, int max_seq_)
        : d_model(d_model_), d_head(d_head_), max_seq(max_seq_),
          Wq(d_model_ * d_head_),
          Wk(d_model_ * d_head_),
          Wv(d_model_ * d_head_) {}

    // X: [T * d_model]
    // Y: [T * d_head]
    void forward(const std::vector<float>& X, int T, std::vector<float>& Y) const;
};
