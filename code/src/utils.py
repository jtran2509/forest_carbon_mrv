import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import compute_iou

def prepare_data_loader(image_paths, mask_paths, batch_size=16, train_split=0.7, val_split=0.15, test_split=0.15):
    dataset_size = len(image_paths)
    train_len = int(dataset_size * train_split)
    val_len = int(dataset_size * val_split)
    test_len = dataset_size - train_len - val_len
    
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]
    
    from dataset import ForestCarbonDataset, get_train_transform
    
    train_ds = ForestCarbonDataset(
        [image_paths[i] for i in train_indices],
        [mask_paths[i] for i in train_indices],
        transform=get_train_transform()
    )
    val_ds = ForestCarbonDataset(
        [image_paths[i] for i in val_indices],
        [mask_paths[i] for i in val_indices],
        transform=None
    )
    test_ds = ForestCarbonDataset(
        [image_paths[i] for i in test_indices],
        [mask_paths[i] for i in test_indices],
        transform=None
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def compute_dice(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_iou = []
    all_dice = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Tính IoU
            iou = compute_iou(y_pred=preds, y=labels, ignore_empty=False)
            all_iou.append(torch.mean(iou).item())
            
            # Tính Dice thủ công
            for i in range(preds.shape[0]):
                dice = compute_dice(preds[i], labels[i])
                all_dice.append(dice)
    
    return np.mean(all_iou), np.mean(all_dice)
