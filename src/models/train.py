import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss #Combination of Dice and Cross Entropy
from pathlib import Path
from src.models.attention_unet import AttentionUNet
from src.models.dataset import ForestCarbonDataset, get_train_transform

# Configuration and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

def train_model():
    # Setup Data 
    # Fetch the lists from src/data/preprocess.py results
    train_images = ['path to images1.tiff', '...']
    train_masks = ['path to mask1.tiff', '...']

    train_ds = ForestCarbonDataset(
        image_files = train_images,
        mask_files=train_masks,
        transform=get_train_transform()
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, Loss, and Optimizer
    # img_ch=4 because we use RGB + NIR
    model = AttentionUNet(img_ch = 4, output_ch=1).to(device)

    # DiceCELoss for forest segmentation
    loss_function = DiceCELoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"Starting training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_data in train_loader:
            inputs = batch_data['image'].to(device)
            labels= batch_data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 ==0:
            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / f"attention_unet_epoch_{epoch+1}.pth")

    print("Training complete!")

def get_iou(y_pred, y_true, threshold=0.5):
    """
    Calculate the Intersection over Union (IoU) metric.
    """
    y_pred = (y_pred > threshold).float() # Binary predictions
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection

    # Avoid division by zero
    if union == 0:
        return 1.0
    return (intersection / union).item()

def get_dice_score(y_pred, y_true, threshold=0.5):
    """
    Calculate the Dice Coefficient (F1-score for pixels)
    """
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum()
    total = y_pred.sum() + y_true.sum()

    if total == 0:
        return 1.0
    return (2. * intersection / total).item()


if __name__ == "__main__":
    train_model()
