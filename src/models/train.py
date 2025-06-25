import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

from dataset import BurnAreaDataset
from unet_model import UNet
from utils import check_accuracy

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "best_model.pth"

# Define paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training_patches', 'images')
MASK_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training_patches', 'masks')

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """Performs one full training pass over the dataset."""
    loop = tqdm(loader, desc='Training')

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop postfix
        loop.set_postfix(loss=loss.item())

def main():
    print(f"Using device: {DEVICE}")

    # 1. Create the full dataset
    full_dataset = BurnAreaDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)

    # 2. Split dataset into training and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.15)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Found {dataset_size} total patches.")
    print(f"Training with {len(train_dataset)} patches.")
    print(f"Validating with {len(val_dataset)} patches.")

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialize Model, Optimizer, and Loss Function
    model = UNet(n_channels=5, n_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_val_score = -1.0
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_one_epoch(train_loader, model, optimizer, criterion, DEVICE)
        
        # Validation
        val_score = check_accuracy(val_loader, model, device=DEVICE)
        print(f"Validation Dice Score: {val_score:.4f}")

        # Save the best model
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with Dice Score: {best_val_score:.4f}")

    print("\nTraining complete.")
    print(f"Best validation Dice score: {best_val_score:.4f}")
    print(f"Best model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
