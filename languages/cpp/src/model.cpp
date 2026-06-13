#include <iostream>
#include <vector>
#include <cmath>
#include "model.h"
#include "attention.h"
#include "layernorm.h"
#include "ffn.h"

// ------------------------------------------------------------
// Forward pass: embedding → ln1 → attn → ln2 → ffn
// ------------------------------------------------------------
void Model::forward(const std::vector<int>& tokens, int T, std::vector<float>& out) const {
    out.assign(T * d_model, 0.0f);

    // 1. Token embedding lookup
    for (int t = 0; t < T; ++t) {
        int tok = tokens[t];
        const float* emb = &embedding[tok * d_model];
        float* dst = &out[t * d_model];
        for (int i = 0; i < d_model; ++i) {
            dst[i] = emb[i];
        }
    }

    // 2. LayerNorm 1
    ln1.forward(out, T, out);

    // 3. Self-attention
    attn.forward(out, T, out);

    // 4. LayerNorm 2
    ln2.forward(out, T, out);

    // 5. Feed-forward network
    ffn.forward(out, T, out);
}

// ------------------------------------------------------------
// Output projection: hidden → logits
// logits: [T * vocab_size]
// ------------------------------------------------------------
void Model::project_logits(const std::vector<float>& out, int T, std::vector<float>& logits) const {
    logits.assign(T * vocab_size, 0.0f);

    for (int t = 0; t < T; ++t) {
        const float* h = &out[t * d_model];
        float* logit_row = &logits[t * vocab_size];

        for (int v = 0; v < vocab_size; ++v) {
            const float* w = &Wout[v * d_model];
            float sum = bout[v];
            for (int i = 0; i < d_model; ++i) {
                sum += h[i] * w[i];
            }
            logit_row[v] = sum;
        }
    }
}

// ------------------------------------------------------------
// Autoregressive generation loop
// ------------------------------------------------------------
void generate(Model& model, std::vector<int>& tokens, int max_new) {
    for (int step = 0; step < max_new; ++step) {
        int T = (int)tokens.size();

        std::vector<float> hidden;
        model.forward(tokens, T, hidden);

        std::vector<float> logits;
        model.project_logits(hidden, T, logits);

        // Greedy decode: pick argmax of last timestep
        const float* last = &logits[(T - 1) * model.vocab_size];
        int best = 0;
        float best_val = last[0];

        for (int v = 1; v < model.vocab_size; ++v) {
            if (last[v] > best_val) {
                best_val = last[v];
                best = v;
            }
        }

        tokens.push_back(best);
    }
}

// ------------------------------------------------------------
// Introspection utilities
// ------------------------------------------------------------
void Model::print_structure() const {
    std::cout << "Model structure:\n";
    std::cout << "  d_model:     " << d_model << "\n";
    std::cout << "  d_head:      " << d_head << "\n";
    std::cout << "  max_seq:     " << max_seq << "\n";
    std::cout << "  vocab_size:  " << vocab_size << "\n";
}

void Model::dump_weights() const {
    std::cout << "Weight summary:\n";
    std::cout << "  embedding: " << embedding.size() << " floats\n";
    std::cout << "  Wout:      " << Wout.size()      << " floats\n";
    std::cout << "  bout:      " << bout.size()      << " floats\n";
}
