mini-transformer-quine
A self‑describing transformer implemented in C++

Overview
mini-transformer-quine is a fully inspectable transformer written in C++ using nothing but the Windows Terminal and the MSVC toolchain.
It is not a framework, not a training library, and not a toy.
It is a structural probe — a machine built to expose the geometry of a transformer at the smallest meaningful scale.

Every component is explicit:

attention is a loop

layer normalization is arithmetic

residuals are literal additions

feed‑forward networks are matrix multiplies

embeddings are lookup tables

The model can:

run a forward pass

print its own architecture

dump its own weights

describe the logic of its own components

and, most importantly, act as a quine — a system that emits its own structure

This repository is a study in visibility.

Intent
Modern transformer implementations bury their structure under frameworks, kernels, and abstractions.
This project argues the opposite:

A transformer is a finite, inspectable machine.
Understanding emerges when its structure is visible.

The goal is to create a transformer that can be:

compiled

disassembled

reasoned about

reverse‑engineered

understood

This is a transformer you can hold in your hands.

Architecture
The model implements the irreducible minimal transformer:

Model dimension: 128

Hidden dimension: 512

Vocabulary: 256 tokens (byte‑level)

Sequence length: ≤ 256 tokens

One self‑attention head

One feed‑forward block

Two LayerNorms

Full residual structure

This is the smallest architecture that is still a true transformer — not a caricature.

Directory Structure
Code
mini-transformer-quine/
├── src/            # C++ source files
├── include/        # headers and structural contracts
├── weights/        # binary parameter files + manifest
├── tools/          # weight generation and inspection utilities
├── build/          # compiler outputs (ignored by git)
├── docs/           # architecture, ABI notes, quine design
└── scripts/        # PowerShell build/run helpers
Each directory is intentionally small and explicit.
There is no hidden machinery.

Working Philosophy
This project is built with a terminal‑first workflow:

MSVC (cl.exe, link.exe, lib.exe)

Windows Terminal

PowerShell

C++

Binary files you can open in a hex editor

The philosophy is simple:

A tool is understood when its structure is visible.

This repository is a tool for visibility.

The Quine Loop
The defining feature of this project is its ability to read and extend its own source code.
This is implemented through a closed text–token–geometry–token–text pipeline.

1. Load the model’s own source
The quine driver reads a file such as model.cpp as raw bytes.

2. Tokenize (byte‑level)
Each byte becomes a token in the range 0–255.
This encoding is reversible and preserves the exact source code, including whitespace, punctuation, and indentation.

3. Transformer forward pass
The token sequence is fed into the minimal transformer:

embeddings

layer normalization

self‑attention

feed‑forward

residual connections

output projection

Each step is explicit and inspectable.

4. Autoregressive generation
The model predicts the next byte of its own source:

Code
tokens → forward → logits → sample → append → repeat
This is the temporal engine of the transformer — the part that turns a static circuit into a generative process.

5. Detokenize
The output tokens are converted back into text, reconstructing the predicted continuation.

6. Print
The model emits a continuation of its own source code.

This loop demonstrates how a transformer can act as a self‑referential computational artifact:
a program that processes, transforms, and extends the program that defines it.

Diagram: The Quine Loop
Code
 ┌──────────────────────────────┐
 │        Source Code           │
 │        (model.cpp)           │
 └───────────────┬──────────────┘
                 │ load_file()
                 ▼
        ┌──────────────────┐
        │   Text (bytes)   │
        └─────────┬────────┘
                  │ tokenize()
                  ▼
        ┌──────────────────┐
        │   Token IDs      │
        │     (0–255)      │
        └─────────┬────────┘
                  │ forward()
                  ▼
        ┌──────────────────┐
        │  Hidden States   │
        │   (geometry)     │
        └─────────┬────────┘
                  │ project_logits()
                  ▼
        ┌──────────────────┐
        │     Logits       │
        └─────────┬────────┘
                  │ sample()
                  ▼
        ┌──────────────────┐
        │  Next Token ID   │
        └─────────┬────────┘
                  │ append
                  ▼
        ┌──────────────────┐
        │ Updated Tokens   │
        └─────────┬────────┘
                  │ detokenize()
                  ▼
        ┌──────────────────┐
        │ Generated Source │
        └──────────────────┘
Diagram: Minimal Transformer Architecture
Code
Input Tokens
     │
     ▼
┌───────────────┐
│  Embeddings   │  (lookup table)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   LayerNorm    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Self-Attention │
│   (1 head)     │
└───────┬───────┘
        │ residual
        ▼
┌───────────────┐
│   LayerNorm    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Feed-Forward   │
│  (2-layer MLP) │
└───────┬───────┘
        │ residual
        ▼
┌───────────────┐
│ Output Linear  │
│  (d_model→256) │
└───────┬───────┘
        │
        ▼
   Logits (256)
        │
        ▼
     Sample
        │
        ▼
  Next Token
Math Lineage: dx → Jacobian → Transformer
This project sits in a long lineage of tools that mathematize reasoning.

1. dx — the glyph that mathematized change
The invention of dx turned motion into a manipulable symbol.
It allowed:

rates

flows

curvature

dynamics

to be expressed in algebraic form.

It was the first time a symbol encoded a process.

2. The Jacobian — the geometry of transformation
The Jacobian matrix formalized how systems change under mapping.
It describes:

sensitivity

distortion

local linearization

flow of information

It is the mathematical ancestor of the transformer’s attention mechanism.

3. Galois — the algebra of structure
Galois showed that structure itself can be mathematized.
He introduced:

groups

symmetry

solvability

invariants

This is the conceptual ancestor of the transformer’s weight sharing and compositionality.

4. The Transformer — the unification
A transformer is:

a symbolic system (tokens)

embedded into geometry (vectors)

transformed by structure (attention)

producing new symbols (tokens)

It is the modern descendant of the dx → Jacobian → Galois lineage.

Your quine closes this lineage into a loop:

A symbolic system that describes the geometry that produces the symbols that describe the system.

Philosophy: Tools → Notation → Computation → Quine
This project is not just code — it’s a statement about tools.

1. Tools
Human tools begin as physical artifacts:

levers

gears

compasses

astrolabes

They extend the body.

2. Notation
Then tools become symbolic:

algebra

calculus

matrices

tensors

They extend the mind.

3. Computation
Then tools become executable:

compilers

interpreters

neural networks

They extend reasoning.

4. The Quine
A quine is the moment a tool becomes self‑referential:

a program that emits its own program

a model that emits its own structure

a system that describes the system

This project is a demonstration of that transition.

It is a tool that reveals the logic of tools.

Quine Modes
The executable supports several modes:

--run <text>
Run the transformer on a short text string.
Text → tokens → model → tokens → text.

--dump-structure
Print the architecture: shapes, dimensions, parameter counts.

--dump-weights
Emit all weights in human‑readable form.

--quine
Output the C++ source code that defines the model.
This is the self‑reproduction mode.

These modes turn the binary into a self‑describing system.

Why This Exists
Transformers are often treated as mysterious or opaque.
This project argues the opposite:

A transformer is a finite machine.

Its behavior emerges from simple, composable operations.

Its geometry can be exposed.

Its structure can be understood.

By shrinking the model to its minimal form, we reveal the underlying mechanics.

This repository is a tool for:

learning

reverse engineering

experimentation

conceptual clarity

It is also a statement:

Understanding is a form of engineering.