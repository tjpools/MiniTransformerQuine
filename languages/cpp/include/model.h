#pragma once
#include <vector>
#include "attention.h"
#include "layernorm.h"
#include "ffn.h"
//***************************************************************************
struct Model {
	// Core hyperparameters: 4 invariants that define the geometry of the block
    int d_model;
    int d_head;
    int max_seq;
    int vocab_size;

    // Members ordered to match the forward pass visually
	// Weights and submodules
    std::vector<float> embedding;
    LayerNorm ln1;
    AttentionHead attn;
    LayerNorm ln2;
    FeedForward ffn;
    std::vector<float> Wout; // [d_model * vocab_size]
    std::vector<float> bout; // [vocab_size]
	// Introspection methods
	void print_structure() const;
	void dump_weights() const;


    // C++ initializes members in the order they are declared
    Model(int d_model_, int d_head_, int max_seq_, int vocab_size_)
        : d_model(d_model_),
          d_head(d_head_),
          max_seq(max_seq_),
          vocab_size(vocab_size_),
          embedding(vocab_size_ * d_model_),
          ln1(d_model_),
          attn(d_model_, d_head_, max_seq_),
          ln2(d_model_),
          ffn(d_model_, 4 * d_model_),
          Wout(d_model_ * vocab_size_),
          bout(vocab_size_)
    {}

    // tokens: [T]
    // output: [T * d_model]
    void forward(const std::vector<int>& tokens, int T, std::vector<float>& out) const;

    // output projection: hidden → logits
    void project_logits(const std::vector<float>& out, int T, std::vector<float>& logits) const;
};

	// Autoregressive generation loop
	void generate(Model& model, std::vector<int>& tokens, int max_new);
