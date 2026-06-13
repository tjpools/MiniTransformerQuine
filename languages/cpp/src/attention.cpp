#include "attention.h"
#include <cmath>
#include <algorithm>

static void matmul(
    const std::vector<float>& A, // [T * d_model]
    const std::vector<float>& W, // [d_model * d_head]
    int T, int d_model, int d_head,
    std::vector<float>& out      // [T * d_head]
) {
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < d_head; j++) {
            float sum = 0.0f;
            for (int i = 0; i < d_model; i++) {
                sum += A[t * d_model + i] * W[i * d_head + j];
            }
            out[t * d_head + j] = sum;
        }
    }
}

void AttentionHead::forward(
    const std::vector<float>& X, // [T * d_model]
    int T,
    std::vector<float>& Y        // [T * d_head]
) const {
    // Temporary buffers
    std::vector<float> Q(T * d_head);
    std::vector<float> K(T * d_head);
    std::vector<float> V(T * d_head);
    std::vector<float> scores(T * T);

    // 1. Project to Q, K, V
    matmul(X, Wq, T, d_model, d_head, Q);
    matmul(X, Wk, T, d_model, d_head, K);
    matmul(X, Wv, T, d_model, d_head, V);

    // 2. Compute scaled dot-product attention scores
    float scale = 1.0f / std::sqrt(float(d_head));
    for (int t = 0; t < T; t++) {
        for (int u = 0; u < T; u++) {
            float dot = 0.0f;
            for (int j = 0; j < d_head; j++) {
                dot += Q[t * d_head + j] * K[u * d_head + j];
            }
            scores[t * T + u] = dot * scale;
        }
    }

    // 3. Softmax each row of scores
    for (int t = 0; t < T; t++) {
        float maxv = -1e30f;
        for (int u = 0; u < T; u++)
            maxv = std::max(maxv, scores[t * T + u]);

        float sum = 0.0f;
        for (int u = 0; u < T; u++) {
            float e = std::exp(scores[t * T + u] * 1.0f - maxv);
            scores[t * T + u] = e;
            sum += e;
        }
        for (int u = 0; u < T; u++)
            scores[t * T + u] /= sum;
    }

    // 4. Weighted sum of V
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < d_head; j++) {
            float sum = 0.0f;
            for (int u = 0; u < T; u++) {
                sum += scores[t * T + u] * V[u * d_head + j];
            }
            Y[t * d_head + j] = sum;
        }
    }
}
