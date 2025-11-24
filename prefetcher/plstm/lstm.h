#ifndef PLSTM_LSTM_H
#define PLSTM_LSTM_H

#include <math.h>
#include <stdint.h>
#include <string>
#include <cstring>
#include "cnpy.h"

#define LSTM_INPUT_SIZE 4
#define LSTM_HIDDEN_SIZE 128
#define LSTM_OUTPUT_SIZE 1


template <typename T>
class LSTM {
public:
    T W_ih[LSTM_INPUT_SIZE][4 * LSTM_HIDDEN_SIZE];
    T W_hh[LSTM_HIDDEN_SIZE][4 * LSTM_HIDDEN_SIZE];
    T W_out[LSTM_OUTPUT_SIZE][LSTM_HIDDEN_SIZE];

    T b_ih[4 * LSTM_HIDDEN_SIZE];
    T b_hh[4 * LSTM_HIDDEN_SIZE];
    T b_out[LSTM_OUTPUT_SIZE];

    T h[LSTM_HIDDEN_SIZE];     // hidden state
    T c[LSTM_HIDDEN_SIZE];     // cell state

    LSTM() {
        for(int i=0;i<LSTM_HIDDEN_SIZE;i++){
            h[i] = 0.0;
            c[i] = 0.0;
        }
    }
    static inline T sigmoid(T x) { return 1.0 / (1.0 + exp(-x)); }

    void load_npy_matrix(const std::string& path, T* dst, size_t rows, size_t cols)
    {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        T* data = arr.data<T>();

        if (arr.shape.size() != 2 ||
            arr.shape[1] != rows ||
            arr.shape[0] != cols)
        {
            std::cout<<"W_ih dimensions: "<<LSTM_INPUT_SIZE<<"x"<<4*LSTM_HIDDEN_SIZE<<'\n';
            std::cout<<"Array dimension: "<<arr.shape[0]<<"x"<<arr.shape[1]<<std::endl;
            throw std::runtime_error("Shape mismatch in " + path);
        }

        memcpy(dst, data, sizeof(T) * rows * cols);
    }

    void load_npy_vector(const std::string& path, T* dst, size_t count)
    {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        T* data = arr.data<T>();

        if (arr.shape.size() != 1 ||
            arr.shape[0] != count)
        {
            throw std::runtime_error("Shape mismatch in " + path);
        }

        memcpy(dst, data, sizeof(T) * count);
    }

    void initialise_from_folder(const std::string folder) {
        load_npy_matrix(folder + "/W_ih.npy",
                        &W_ih[0][0],
                        LSTM_INPUT_SIZE,
                        4 * LSTM_HIDDEN_SIZE);

        load_npy_matrix(folder + "/W_hh.npy",
                        &W_hh[0][0],
                        LSTM_HIDDEN_SIZE,
                        4 * LSTM_HIDDEN_SIZE);

        load_npy_vector(folder + "/b_ih.npy",
                        b_ih,
                        4 * LSTM_HIDDEN_SIZE);

        load_npy_vector(folder + "/b_hh.npy",
                        b_hh,
                        4 * LSTM_HIDDEN_SIZE);

        load_npy_vector(folder + "/b_out.npy",
                        b_out,
                        LSTM_OUTPUT_SIZE);

        load_npy_matrix(folder + "/W_out.npy",
                        &W_out[0][0],
                        LSTM_HIDDEN_SIZE,
                        LSTM_OUTPUT_SIZE);
        
    }

    void predict(T *x, T out[LSTM_OUTPUT_SIZE]) {

        T gates[4 * LSTM_HIDDEN_SIZE];

        // ---------- 1. Compute gates (W_ih x + b_ih) ----------
        for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
            gates[g] = b_ih[g];

        for (int j = 0; j < LSTM_INPUT_SIZE; j++) {
            T v = x[j];
            for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
                gates[g] += v * W_ih[j][g];
        }

        // ---------- 2. Add recurrent part (W_hh h + b_hh) ----------
        for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
            gates[g] += b_hh[g];

        for (int j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            T v = h[j];
            for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
                gates[g] += v * W_hh[j][g];
        }

        // Split gates into i, f, g, o
        T* i_t = gates;
        T* f_t = gates + LSTM_HIDDEN_SIZE;
        T* g_t = gates + 2 * LSTM_HIDDEN_SIZE;
        T* o_t = gates + 3 * LSTM_HIDDEN_SIZE;

        // ---------- 3. Apply activations & update cell ----------
        for (int k = 0; k < LSTM_HIDDEN_SIZE; k++) {
            i_t[k] = sigmoid(i_t[k]);
            f_t[k] = sigmoid(f_t[k]);
            g_t[k] = tanh(g_t[k]);
            o_t[k] = sigmoid(o_t[k]);

            c[k] = f_t[k] * c[k] + i_t[k] * g_t[k];
            h[k] = o_t[k] * tanh(c[k]);
        }

        // ---------- 4. Output hidden ----------
        for (int i = 0; i < LSTM_OUTPUT_SIZE; i++) {
            T sum = 0.0;
            for (int j = 0; j < LSTM_HIDDEN_SIZE; j++) {
                sum += W_out[i][j] * h[j];
            }
            out[i] = sum + b_out[i];
        }
    }
};

#endif
