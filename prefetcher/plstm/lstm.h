#ifndef PLSTM_LSTM_H
#define PLSTM_LSTM_H

#include <math.h>
#include <string>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include "cnpy.h"

// ── Default network dimensions (override before #include if needed) ───────────
#ifndef LSTM_INPUT_SIZE
#  define LSTM_INPUT_SIZE  4
#endif
#ifndef LSTM_HIDDEN_SIZE
#  define LSTM_HIDDEN_SIZE 128
#endif
#ifndef LSTM_OUTPUT_SIZE
#  define LSTM_OUTPUT_SIZE 1
#endif
#ifndef LSTM_NUM_LAYERS
#  define LSTM_NUM_LAYERS  2
#endif

// ─────────────────────────────────────────────────────────────────────────────
// LSTMLayer — pure compute struct, no file I/O
//
//   Owns weights, biases, and recurrent state for one layer.
//   Weights are written directly by MultiLayerLSTM at startup; this struct
//   never touches the filesystem.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T, int IN, int H>
struct LSTMLayer {
    T W_ih[IN][4 * H];
    T W_hh[H ][4 * H];
    T b_ih[4 * H];
    T b_hh[4 * H];

    T h[H];   // hidden state
    T c[H];   // cell state

    LSTMLayer() { reset(); }

    void reset() {
        for (int i = 0; i < H; i++) { h[i] = T(0); c[i] = T(0); }
    }

    // One time-step forward pass.  Result is left in h[].
    void step(const T* x) {
        T gates[4 * H];

        for (int g = 0; g < 4 * H; g++)
            gates[g] = b_ih[g] + b_hh[g];

        for (int j = 0; j < IN; j++) {
            T v = x[j];
            for (int g = 0; g < 4 * H; g++)
                gates[g] += v * W_ih[j][g];
        }

        for (int j = 0; j < H; j++) {
            T v = h[j];
            for (int g = 0; g < 4 * H; g++)
                gates[g] += v * W_hh[j][g];
        }

        T* i_t = gates;
        T* f_t = gates +     H;
        T* g_t = gates + 2 * H;
        T* o_t = gates + 3 * H;

        for (int k = 0; k < H; k++) {
            i_t[k] = sigmoid(i_t[k]);
            f_t[k] = sigmoid(f_t[k]);
            g_t[k] = tanh   (g_t[k]);
            o_t[k] = sigmoid(o_t[k]);

            c[k] = f_t[k] * c[k] + i_t[k] * g_t[k];
            h[k] = o_t[k] * tanh(c[k]);
        }
    }

private:
    static inline T sigmoid(T x) { return T(1) / (T(1) + exp(-x)); }
};


// ─────────────────────────────────────────────────────────────────────────────
// MultiLayerLSTM — owns all layers and is the sole entry point for file I/O
//
//   initialise_from_folder() opens every .npy file exactly once, in sequence,
//   and writes directly into the appropriate layer's weight arrays.
//   After that call, the network is fully self-contained with no further I/O.
//
//   Folder layout:
//     root/layer_0/W_ih.npy  W_hh.npy  b_ih.npy  b_hh.npy
//     root/layer_1/W_ih.npy  ...
//     root/W_out.npy
//     root/b_out.npy
// ─────────────────────────────────────────────────────────────────────────────
template <typename T,
          int NUM_LAYERS,
          int INPUT_SIZE  = LSTM_INPUT_SIZE,
          int HIDDEN_SIZE = LSTM_HIDDEN_SIZE,
          int OUTPUT_SIZE = LSTM_OUTPUT_SIZE>
class MultiLayerLSTM {
    static_assert(NUM_LAYERS >= 1, "MultiLayerLSTM requires at least 1 layer.");

public:
    LSTMLayer<T, INPUT_SIZE,  HIDDEN_SIZE> layer0;
    LSTMLayer<T, HIDDEN_SIZE, HIDDEN_SIZE> layers[NUM_LAYERS > 1 ? NUM_LAYERS - 1 : 1];

    T W_out[OUTPUT_SIZE][HIDDEN_SIZE];
    T b_out[OUTPUT_SIZE];

    // Reset all recurrent states (call between independent sequences)
    void reset() {
        layer0.reset();
        for (int l = 0; l < NUM_LAYERS - 1; l++)
            layers[l].reset();
    }

    // Load all weights in one pass — the only function that touches the disk.
    void initialise_from_folder(const std::string& root) {
        // Layer 0 (input size may differ from hidden size)
        load_matrix(root + "/layer_0/W_ih.npy", &layer0.W_ih[0][0], INPUT_SIZE,  4 * HIDDEN_SIZE);
        load_matrix(root + "/layer_0/W_hh.npy", &layer0.W_hh[0][0], HIDDEN_SIZE, 4 * HIDDEN_SIZE);
        load_vector(root + "/layer_0/b_ih.npy",  layer0.b_ih,                     4 * HIDDEN_SIZE);
        load_vector(root + "/layer_0/b_hh.npy",  layer0.b_hh,                     4 * HIDDEN_SIZE);

        // Layers 1 ... NUM_LAYERS-1
        for (int l = 1; l < NUM_LAYERS; l++) {
            const std::string prefix = root + "/layer_" + std::to_string(l);
            LSTMLayer<T, HIDDEN_SIZE, HIDDEN_SIZE>& ly = layers[l - 1];
            load_matrix(prefix + "/W_ih.npy", &ly.W_ih[0][0], HIDDEN_SIZE, 4 * HIDDEN_SIZE);
            load_matrix(prefix + "/W_hh.npy", &ly.W_hh[0][0], HIDDEN_SIZE, 4 * HIDDEN_SIZE);
            load_vector(prefix + "/b_ih.npy",  ly.b_ih,                    4 * HIDDEN_SIZE);
            load_vector(prefix + "/b_hh.npy",  ly.b_hh,                    4 * HIDDEN_SIZE);
        }

        // Output projection
        load_matrix(root + "/W_out.npy", &W_out[0][0], HIDDEN_SIZE, OUTPUT_SIZE);
        load_vector(root + "/b_out.npy",  b_out,                    OUTPUT_SIZE);
    }

    // One time-step forward pass
    void predict(const T* x, T out[OUTPUT_SIZE]) {
        layer0.step(x);

        const T* prev_h = layer0.h;
        for (int l = 0; l < NUM_LAYERS - 1; l++) {
            layers[l].step(prev_h);
            prev_h = layers[l].h;
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            T sum = b_out[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += W_out[i][j] * prev_h[j];
            out[i] = sum;
        }
    }

private:
    // ── Loading helpers — private, called only from initialise_from_folder ───

    // Expected on-disk shape: [cols, rows]  (PyTorch default row-major export)
    static void load_matrix(const std::string& path, T* dst, int rows, int cols) {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        if (arr.shape.size() != 2 ||
            static_cast<int>(arr.shape[1]) != rows ||
            static_cast<int>(arr.shape[0]) != cols)
        {
            std::cout << "Expected [" << cols << "x" << rows << "]"
                      << "  got ["   << arr.shape[0] << "x" << arr.shape[1] << "]\n";
            throw std::runtime_error("Shape mismatch loading " + path);
        }
        memcpy(dst, arr.data<T>(), sizeof(T) * rows * cols);
    }

    static void load_vector(const std::string& path, T* dst, int count) {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        if (arr.shape.size() != 1 ||
            static_cast<int>(arr.shape[0]) != count)
        {
            throw std::runtime_error("Shape mismatch loading " + path);
        }
        memcpy(dst, arr.data<T>(), sizeof(T) * count);
    }
};


// ── Convenience alias matching the original single-layer API ─────────────────
template <typename T>
using LSTM = MultiLayerLSTM<T,
                             LSTM_NUM_LAYERS,
                             LSTM_INPUT_SIZE,
                             LSTM_HIDDEN_SIZE,
                             LSTM_OUTPUT_SIZE>;

#endif // PLSTM_LSTM_H