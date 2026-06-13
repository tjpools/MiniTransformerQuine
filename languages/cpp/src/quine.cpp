#include "quine.h"
#include "model.h"
#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>

// forward declaration
void generate(Model& model, std::vector<int>& tokens, int max_new);

// ------------------------------------------------------------
// Load a file into a string
// ------------------------------------------------------------
static std::string load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: could not open " << path << "\n";
        return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ------------------------------------------------------------
// Print the model's own source code (quine driver)
// ------------------------------------------------------------
void print_quine_source() {
    // 1. Load source file
    std::string source = load_file("model.cpp");

    // 2. Tokenize
    std::vector<int> tokens = tokenize(source);

    // 3. Construct model
    int d_model = 128;
    int d_head  = 128;
    int max_seq = 256;
    int vocab_size = 256;

    Model model(d_model, d_head, max_seq, vocab_size);

    // 4. Generate continuation
    generate(model, tokens, 200);

    // 5. Detokenize
    std::string out = detokenize(tokens);

    // 6. Print
    std::cout << out;
}
