#include "layernorm.h"
#include <cmath>

void LayerNorm::forward(
    const std::vector<float>& X,
    int T,
    std::vector<float>& Y
) const {
    Y.resize(T * dim);

    for (int t = 0; t < T; t++) {
        const float* x = &X[t * dim];

        // mean
        float mean = 0.0f;
        for (int i = 0; i < dim; i++)
            mean += x[i];
        mean /= dim;

        // variance
        float var = 0.0f;
        for (int i = 0; i < dim; i++) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= dim;

        float inv = 1.0f / std::sqrt(var + eps);

        // normalize + affine
        for (int i = 0; i < dim; i++) {
            float xn = (x[i] - mean) * inv;
            Y[t * dim + i] = xn * gamma[i] + beta[i];
        }
    }
}
