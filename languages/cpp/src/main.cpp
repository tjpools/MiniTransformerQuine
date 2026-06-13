#include <iostream>
#include <string>
#include <vector>
#include "model.h"
#include "tokenizer.h"
#include "quine.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  mini-transformer-quine --run <text>\n"
                  << "  mini-transformer-quine --dump-structure\n"
                  << "  mini-transformer-quine --dump-weights\n"
                  << "  mini-transformer-quine --quine\n";
        return 0;
    }

    std::string mode = argv[1];

    // Tiny demo model hyperparameters
    int d_model = 32;
    int d_head = 4;
    int max_seq = 64;
    int vocab_size = 128;

    Model model(d_model, d_head, max_seq, vocab_size);

    // ------------------------------------------------------------
    // Dump structure
    // ------------------------------------------------------------
    if (mode == "--dump-structure") {
        model.print_structure();
        return 0;
    }

    // ------------------------------------------------------------
    // Dump weights
    // ------------------------------------------------------------
    if (mode == "--dump-weights") {
        model.dump_weights();
        return 0;
    }

    // ------------------------------------------------------------
    // Quine mode
    // ------------------------------------------------------------
    if (mode == "--quine") {
        print_quine_source();
        return 0;
    }

    // ------------------------------------------------------------
    // Run inference
    // ------------------------------------------------------------
    if (mode == "--run") {
        if (argc < 3) {
            std::cerr << "Error: --run requires input text.\n";
            return 1;
        }

        std::string input = argv[2];
        auto tokens = tokenize(input);

        generate(model, tokens, 16);

        std::cout << "Output tokens: ";
        for (int t : tokens) {
            std::cout << t << " ";
        }
        std::cout << "\n";

        return 0;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
