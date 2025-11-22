#ifndef PLSTM_LSTM_H
#define PLSTM_LSTM_H

#include <math.h>
#include <stdint.h>

#define LSTM_INPUT_SIZE 32
#define LSTM_HIDDEN_SIZE 32

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

class LSTM {
public:
    double W_ih[LSTM_INPUT_SIZE][4 * LSTM_HIDDEN_SIZE];
    double W_hh[LSTM_HIDDEN_SIZE][4 * LSTM_HIDDEN_SIZE];

    double b_ih[4 * LSTM_HIDDEN_SIZE];
    double b_hh[4 * LSTM_HIDDEN_SIZE];

    double h[LSTM_HIDDEN_SIZE];     // hidden state
    double c[LSTM_HIDDEN_SIZE];     // cell state

    LSTM() {
        for(int i=0;i<LSTM_HIDDEN_SIZE;i++){
            h[i] = 0.0;
            c[i] = 0.0;
        }
    }

    void predict(const double x[LSTM_INPUT_SIZE], double out[LSTM_HIDDEN_SIZE]) {

        double gates[4 * LSTM_HIDDEN_SIZE];

        // ---------- 1. Compute gates (W_ih x + b_ih) ----------
        for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
            gates[g] = b_ih[g];

        for (int j = 0; j < LSTM_INPUT_SIZE; j++) {
            double v = x[j];
            for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
                gates[g] += v * W_ih[j][g];
        }

        // ---------- 2. Add recurrent part (W_hh h + b_hh) ----------
        for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
            gates[g] += b_hh[g];

        for (int j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            double v = h[j];
            for (int g = 0; g < 4 * LSTM_HIDDEN_SIZE; g++)
                gates[g] += v * W_hh[j][g];
        }

        // Split gates into i, f, g, o
        double* i_t = gates;
        double* f_t = gates + LSTM_HIDDEN_SIZE;
        double* g_t = gates + 2 * LSTM_HIDDEN_SIZE;
        double* o_t = gates + 3 * LSTM_HIDDEN_SIZE;

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
        for(int k=0;k<LSTM_HIDDEN_SIZE;k++)
            out[k] = h[k];
    }
};

#endif
