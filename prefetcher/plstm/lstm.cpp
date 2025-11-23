#include <iostream>
#include "lstm.h"

int main() {
    LSTM l = LSTM();
    l.initialise_from_folder("lstm_training/lstm_params");
    double deltas[LSTM_INPUT_SIZE] = {
    0.0142, -0.0328, 0.0059, 0.0411,
   -0.0274, 0.0187, -0.0493, 0.0221,
    0.0076, -0.0158, 0.0124, -0.0449,
    0.0033, 0.0285, -0.0097, 0.0472,
   -0.0219, 0.0364, -0.0048, 0.0109,
   -0.0391, 0.0247, 0.0028, -0.0305,
    0.0166, 0.0453, -0.0112, 0.0081,
   -0.0267, 0.0199, -0.0435, 0.0064
    };
    double out[LSTM_OUTPUT_SIZE];
    l.predict(deltas, out);
    for (int i=0; i<LSTM_OUTPUT_SIZE; i++) {
        std::cout<<out[i]<<" ";
    }
    std::cout<<std::endl;
}