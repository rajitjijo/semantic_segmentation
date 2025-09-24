import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
from model import UNET

#HYPERPARAMETERS
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

def train(loader, model, optimizer, loss_fn, scalar):

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        #Forward
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        loop.set_postfix(loss=loss.item())


def main():

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



























if __name__ == "__main__":
    pass