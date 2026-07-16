import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ==========================================
# 1. THE DATALOADER (The Conveyor Belt)
# ==========================================
class TaobaoSequenceDataset(Dataset):
    def __init__(self, pkl_file):
        logging.info("Loading sequence data into memory...")
        self.df = pd.read_pickle(pkl_file)

        # We drop sequences that are essentially empty (just padding)
        self.df = self.df[self.df["behavior_seq"].apply(lambda x: sum(x) > 0)]
        self.df.reset_index(drop=True, inplace=True)
        logging.info(f"Total valid sequences loaded: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # INPUTS (X): The first 19 steps
        # We ensure they are integers for the Embedding layers
        cat_seq = torch.tensor(row["category_seq"][:-1], dtype=torch.long)
        beh_seq = torch.tensor(row["behavior_seq"][:-1], dtype=torch.long)

        # TARGET (y): The 20th step (What they actually did next)
        # We subtract 1 so behaviors (1,2,3,4) become classes (0,1,2,3)
        target_beh = torch.tensor(row["behavior_seq"][-1] - 1, dtype=torch.long)

        # If the target is negative (padding), default it to 0 (View) for safety
        target_beh = torch.clamp(target_beh, min=0)

        return cat_seq, beh_seq, target_beh


# ==========================================
# 2. THE NEURAL NETWORK (The Brain)
# ==========================================
class TaobaoBehaviorGRU(nn.Module):
    def __init__(self, num_categories=10000, num_behaviors=5, hidden_size=64):
        super(TaobaoBehaviorGRU, self).__init__()

        # The Dictionaries: Translate numbers into dense mathematical vectors
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=32
        )
        self.beh_embedding = nn.Embedding(
            num_embeddings=num_behaviors, embedding_dim=16
        )

        # The Memory Engine: GRU (Gated Recurrent Unit)
        # It takes the combined item+behavior vectors and learns the timeline
        self.gru = nn.GRU(input_size=32 + 16, hidden_size=hidden_size, batch_first=True)

        # The Output Layer: Squashes the final memory state into 4 probabilities
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, cat_seq, beh_seq):
        # 1. Look up the vectors
        cat_embs = self.cat_embedding(cat_seq)
        beh_embs = self.beh_embedding(beh_seq)

        # 2. Combine them side-by-side
        combined_input = torch.cat([cat_embs, beh_embs], dim=-1)

        # 3. Pass through the timeline memory engine
        gru_out, hidden_state = self.gru(combined_input)

        # 4. Take the very last memory state (representing step 19) and predict step 20
        final_memory = hidden_state[-1]
        output = self.fc(final_memory)
        return output


# ==========================================
# 3. THE TRAINING LOOP (The Teacher)
# ==========================================
def train_model():
    # Detect Apple Silicon GPU (MPS), fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Neural Network will train on: {device.type.upper()}")

    # 1. Setup Data
    dataset = TaobaoSequenceDataset("data/processed/user_sequences.pkl")
    # Process 512 users at the exact same time
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # 2. Setup Model
    model = TaobaoBehaviorGRU().to(device)
    criterion = nn.CrossEntropyLoss()  # Standard loss for multi-class classification
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005
    )  # The mathematical adjuster

    epochs = 3  # How many times we read the entire dataset

    logging.info("Starting Deep Learning Training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (cat_seq, beh_seq, target_beh) in enumerate(dataloader):
            # Move data to the Mac GPU
            cat_seq, beh_seq, target_beh = (
                cat_seq.to(device),
                beh_seq.to(device),
                target_beh.to(device),
            )

            # Step 1: Clear old math
            optimizer.zero_grad()

            # Step 2: Make a guess
            predictions = model(cat_seq, beh_seq)

            # Step 3: Check how wrong it is
            loss = criterion(predictions, target_beh)

            # Step 4: Adjust the brain (Backpropagation)
            loss.backward()
            optimizer.step()

            # Calculate accuracy metrics
            total_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == target_beh).sum().item()
            total_samples += target_beh.size(0)

            if batch_idx % 50 == 0 and batch_idx > 0:
                current_acc = (correct_predictions / total_samples) * 100
                logging.info(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | Current Accuracy: {current_acc:.2f}% | Loss: {loss.item():.4f}"
                )

        final_epoch_acc = (correct_predictions / total_samples) * 100
        logging.info(
            f"=== Epoch {epoch + 1} Complete | Final Accuracy: {final_epoch_acc:.2f}% ==="
        )

    # Save the trained brain
    # Save the trained brain
    model_save_path = "models/taobao_sequence_model.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Success! Deep Learning Model saved as '{model_save_path}'")

if __name__ == "__main__":
    train_model()
