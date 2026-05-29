import os
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import AttentionUNet
from dataset import ForestCarbonDataset, get_train_transform
from utils import prepare_data_loader, evaluate_model

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--checkpoint-dir', type=str, default=os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = '/opt/ml/input/data/training/processed'

    # DEBUG: Tìm tất cả file .npy trước
    all_npy = glob.glob(os.path.join(data_dir, '**/*.npy'), recursive=True)
    print(f"DEBUG: Total .npy files found: {len(all_npy)}")
    for npy in all_npy[:10]:
        print(f"  - {npy}")
    
    # Tìm images và masks
    all_images = glob.glob(os.path.join(data_dir, '**/images/chip_*.npy'), recursive=True)
    all_masks = glob.glob(os.path.join(data_dir, '**/masks/mask_*.npy'), recursive=True)
    print(f"DEBUG: images found: {len(all_images)}, masks found: {len(all_masks)}")
    all_images.sort()
    all_masks.sort()

    print(f"Found {len(all_images)} images, {len(all_masks)} masks")
    assert len(all_images) == len(all_masks)

    train_loader, val_loader, test_loader = prepare_data_loader(
        all_images, all_masks,
        batch_size=args.batch_size,
        train_split=0.7, val_split=0.15, test_split=0.15
    )

    model = AttentionUNet(img_ch=2, output_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = DiceCELoss(sigmoid=True)

    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(args.checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, '*.pth'))
        if checkpoint_files:
            latest = max(checkpoint_files, key=os.path.getctime)
            checkpoint = torch.load(latest)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from checkpoint at epoch {start_epoch}")

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"✅ Saved checkpoint to {checkpoint_path}")

    final_iou, final_dice = evaluate_model(model, test_loader, device)
    print(f"Final IoU: {final_iou:.4f}, Final Dice: {final_dice:.4f}") # Removed emoji and leading newline
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__ == '__main__':
    train()
