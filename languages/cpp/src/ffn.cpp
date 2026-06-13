#include "ffn.h"
#include <cmath>

static inline float gelu(float x) {
    // Approximate GELU (Hendrycks & Gimpel)
    return 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

void FeedForward::forward(
    const std::vector<float>& X, // [T * d_model]
    int T,
    std::vector<float>& Y        // [T * d_model]
) const {
    Y.resize(T * d_model);

    std::vector<float> H(T * d_hidden);

    // 1. Hidden = GELU(X * W1 + b1)
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < d_hidden; j++) {
            float sum = b1[j];
            for (int i = 0; i < d_model; i++) {
                sum += X[t * d_model + i] * W1[i * d_hidden + j];
            }
            H[t * d_hidden + j] = gelu(sum);
        }
    }

    // 2. Output = H * W2 + b2
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < d_model; j++) {
            float sum = b2[j];
            for (int i = 0; i < d_hidden; i++) {
                sum += H[t * d_hidden + i] * W2[i * d_model + j];
            }
            Y[t * d_model + j] = sum;
        }
    }
}
