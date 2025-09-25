import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
from model import UNET
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy
import os, csv

#HYPERPARAMETERS
experiment_name = "first_test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scalar = torch.GradScaler()
image_height = 160
image_width = 240
learning_rate = 1e-4
batch_size = 16
num_epochs = 3
PIN_MEMORY = True
LOAD_MEMORY = True
train_img_dir = "data/train"
train_mask_dir = "data/train_masks"
valid_img_dir = "data/val"
valid_mask_dir = "data/val_masks"
train_save_dir = f"training_runs/{experiment_name}"

def train(loader, model, optimizer, loss_fn, scalar, epoch):

    loop = tqdm(loader,desc=f"Training: Epoch[{epoch + 1}/{num_epochs}]")
    epoch_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        
        data = data.to(device)
        targets = targets.float().to(device)
        #Forward
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


def main():

    os.makedirs(train_save_dir, exist_ok=True)

    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=(0.0,0.0,0.0),
                std=(1.0,1.0,1.0),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=(0.0,0.0,0.0),
                std=(1.0,1.0,1.0),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_laoder = get_loaders(train_dir=train_img_dir, train_mask_dir=train_mask_dir,
                                           val_dir=valid_img_dir, val_mask_dir=valid_mask_dir, 
                                           batch_size=batch_size,train_transform=train_transform,
                                           val_transform=val_transforms)
    
    with open(os.path.join(train_save_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Accuracy", "Dice_Score"])
    
    for epoch in range(num_epochs):

        loss = train(train_loader, model, optimizer, loss_fn, scalar, epoch)
        save_dir = os.path.join(train_save_dir, f"train{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        acc, dice_score = check_accuracy(val_laoder, model, device, save_dir)
        with open(os.path.join(train_save_dir, "metrics.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, loss, acc, dice_score.item()]) # type: ignore


if __name__ == "__main__":
    main()