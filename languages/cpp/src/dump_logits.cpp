#include <iostream>
#include <vector>
#include "model.h"

int main(int argc, char** argv) {
    char ch = 'A';
    if (argc > 1) {
        ch = argv[1][0];
    }

    int token = static_cast<unsigned char>(ch);

    Model model(32, 4, 64, 256);

    // Prepare token sequence of length 1
    std::vector<int> tokens = { token };

    // Forward pass: hidden state [1 * d_model]
    std::vector<float> hidden(model.d_model);
    model.forward(tokens, 1, hidden);

    // Project to logits [1 * vocab_size]
    std::vector<float> logits(model.vocab_size);
    model.project_logits(hidden, 1, logits);

    // Print logits
    for (int i = 0; i < model.vocab_size; i++) {
        printf("%3d: %.9f\n", i, logits[i]);
    }

    return 0;
}
