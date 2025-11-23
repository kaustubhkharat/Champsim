import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# PARAMETERS
# -----------------------------
STRIDES_DIR = "strides/"
SEQ_LEN = 32
INPUT_SIZE = 32
HIDDEN_SIZE = 128   # increased hidden size
OUTPUT_SIZE = 1     # predict 1 stride at a time for stability
BATCH_SIZE = 64
EPOCHS = 10
CLIP_VALUE = 5      # clip normalized strides to [-5,5]
USE_LOG_TRANSFORM = True
LR = 1e-3
GRAD_CLIP = 1.0     # gradient clipping
ACCURACY_TOL = 0.1  # tolerance for stride prediction "accuracy"

# -----------------------------
# 1. LOAD & FILTER STRIDES
# -----------------------------
def load_stride_sequences():
    sequences = []

    for file in os.listdir(STRIDES_DIR):
        if not file.endswith(".npy"):
            continue

        arr = np.load(os.path.join(STRIDES_DIR, file))
        if len(arr) == 0:
            continue

        # Remove extreme outliers (98th percentile)
        p98 = np.percentile(np.abs(arr), 98)
        arr = arr[np.abs(arr) <= p98]

        if len(arr) > SEQ_LEN + OUTPUT_SIZE:
            sequences.append(arr)

    return sequences

# -----------------------------
# 2. DATASET
# -----------------------------
class StrideDataset(Dataset):
    def __init__(self, sequences, seq_len=SEQ_LEN):
        self.data = []

        # Flatten all sequences into (X,Y) pairs
        all_values = []
        for seq in sequences:
            if USE_LOG_TRANSFORM:
                seq = np.sign(seq) * np.log1p(np.abs(seq))
            all_values.extend(seq)

            for i in range(len(seq) - seq_len - OUTPUT_SIZE):
                x = seq[i : i + seq_len]
                y = seq[i + seq_len : i + seq_len + OUTPUT_SIZE]
                self.data.append((x, y))

        # Compute normalization parameters
        all_values = np.array(all_values)
        self.mean = np.mean(all_values)
        self.std = np.std(all_values) + 1e-6

        print(f"Loaded {len(self.data)} training samples. Mean={self.mean:.4f}, Std={self.std:.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        # Normalize and clip
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x = np.clip(x, -CLIP_VALUE, CLIP_VALUE)
        y = np.clip(y, -CLIP_VALUE, CLIP_VALUE)

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor

# -----------------------------
# 3. LSTM MODEL
# -----------------------------
class LSTMStrideModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------
# 4. SAVE WEIGHTS
# -----------------------------
def save_lstm_weights(model):
    os.makedirs("lstm_params", exist_ok=True)
    lstm = model.lstm

    W_ih = lstm.weight_ih_l0.detach().cpu().numpy()
    W_hh = lstm.weight_hh_l0.detach().cpu().numpy()
    b_ih = lstm.bias_ih_l0.detach().cpu().numpy()
    b_hh = lstm.bias_hh_l0.detach().cpu().numpy()

    W_out = model.fc.weight.detach().cpu().numpy()
    b_out = model.fc.bias.detach().cpu().numpy()

    np.save("lstm_params/W_ih.npy", W_ih)
    np.save("lstm_params/W_hh.npy", W_hh)
    np.save("lstm_params/b_ih.npy", b_ih)
    np.save("lstm_params/b_hh.npy", b_hh)
    np.save("lstm_params/W_out.npy", W_out)
    np.save("lstm_params/b_out.npy", b_out)

    print("Saved all weights to lstm_params/")

# -----------------------------
# 5. TRAINING SCRIPT
# -----------------------------
def main():
    sequences = load_stride_sequences()
    dataset = StrideDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMStrideModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()  # Huber loss for robustness

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X, Y in loader:
            # Expand input to match INPUT_SIZE
            X = X.repeat(1, 1, INPUT_SIZE)

            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * X.size(0)

            # Accuracy: prediction within tolerance
            total_correct += ((torch.abs(pred - Y) < ACCURACY_TOL).sum().item())
            total_samples += Y.numel()

        avg_loss = total_loss / len(dataset)
        accuracy = total_correct / total_samples * 100

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy (±{ACCURACY_TOL}): {accuracy:.2f}%")

    save_lstm_weights(model)

if __name__ == "__main__":
    main()
