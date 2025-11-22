import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

STRIDES_DIR = "strides/"
MAX_DELTA = 1 << 20     # filter out crazy strides (≈ 1M)
SEQ_LEN = 32
INPUT_SIZE = 32
HIDDEN_SIZE = 64
OUTPUT_SIZE = 8
BATCH_SIZE = 64
EPOCHS = 10


# ---------------------------------------------
# 1. LOAD & FILTER STRIDES
# ---------------------------------------------
def load_stride_sequences():
    sequences = []

    for file in os.listdir(STRIDES_DIR):
        if not file.endswith(".npy"):
            continue

        arr = np.load(os.path.join(STRIDES_DIR, file))

        if len(arr) == 0:
            continue

        # ---- NEW FILTER: Remove values above 98th percentile ----
        p98 = np.percentile(np.abs(arr), 98)
        arr = arr[np.abs(arr) <= p98]

        # Need at least SEQ_LEN + OUTPUT_SIZE
        if len(arr) > SEQ_LEN + OUTPUT_SIZE:
            sequences.append(arr)

    return sequences



# ---------------------------------------------
# 2. DATASET
# ---------------------------------------------
class StrideDataset(Dataset):
    def __init__(self, sequences, seq_len=SEQ_LEN):
        self.data = []
        for seq in sequences:
            for i in range(len(seq) - seq_len - OUTPUT_SIZE):
                x = seq[i : i + seq_len]
                y = seq[i + seq_len : i + seq_len + OUTPUT_SIZE]
                self.data.append((x, y))

        # Normalize
        self.mean = np.mean([x for (x, _) in self.data])
        self.std = np.std([x for (x, _) in self.data]) + 1e-6

        print(f"Loaded {len(self.data)} training samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        # LSTM expects (seq_len, input_size)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


# ---------------------------------------------
# 3. LSTM MODEL
# ---------------------------------------------
class LSTMStrideModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last hidden state
        return out


# ---------------------------------------------
# 4. SAVE WEIGHTS INTO SEPARATE FILES
# ---------------------------------------------
def save_lstm_weights(model):
    lstm = model.lstm

    # torch names
    W_ih = lstm.weight_ih_l0.detach().cpu().numpy()
    W_hh = lstm.weight_hh_l0.detach().cpu().numpy()
    b_ih = lstm.bias_ih_l0.detach().cpu().numpy()
    b_hh = lstm.bias_hh_l0.detach().cpu().numpy()

    W_out = model.fc.weight.detach().cpu().numpy()
    b_out = model.fc.bias.detach().cpu().numpy()

    # save separately
    np.save("lstm_params/W_ih.npy", W_ih)
    np.save("lstm_params/W_hh.npy", W_hh)
    np.save("lstm_params/b_ih.npy", b_ih)
    np.save("lstm_params/b_hh.npy", b_hh)
    np.save("lstm_params/W_out.npy", W_out)
    np.save("lstm_params/b_out.npy", b_out)

    print("Saved all weights.")


# ---------------------------------------------
# 5. TRAINING SCRIPT
# ---------------------------------------------
def main():
    sequences = load_stride_sequences()
    dataset = StrideDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMStrideModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for X, Y in loader:
            # Expand input to match input_size = 32
            X = X.repeat(1, 1, INPUT_SIZE)

            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    save_lstm_weights(model)


if __name__ == "__main__":
    main()
