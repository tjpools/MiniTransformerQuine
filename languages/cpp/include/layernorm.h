#pragma once
#include <vector>

struct LayerNorm {
    int dim;
    float eps;

    std::vector<float> gamma; // scale
    std::vector<float> beta;  // shift

    LayerNorm(int dim_, float eps_ = 1e-5f)
        : dim(dim_), eps(eps_),
          gamma(dim_, 1.0f),
          beta(dim_, 0.0f) {}

    // X: [T * dim]
    // Y: [T * dim]
    void forward(const std::vector<float>& X, int T, std::vector<float>& Y) const;
};
