import torch, torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import os
import random

def save_checkpoint(state, filename="my_checkpoint.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True
):
    train_ds = CarvanaDataset(train_dir, train_mask_dir, train_transform)
    train_loader = DataLoader(train_ds, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = CarvanaDataset(val_dir, val_mask_dir, val_transform)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def check_accuracy(loader, model, device, save_dir):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    loop = tqdm(loader, desc="Validating")
    save_batch = random.choice(range(len(loader)))

    with torch.no_grad():

        for idx, (x, y) in enumerate(loop):

            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds==y).sum().item()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum() + 1e-8)
            if idx == save_batch:
                save_predictions_as_imgs(x,y,preds,os.path.join(save_dir, "sample.png"))

            loop.set_postfix(acc=(num_correct/num_pixels), dice_score=(dice_score/(idx+1)).item())

    acc = num_correct / num_pixels
    dice = dice_score / len(loader) 

    return acc, dice

def save_predictions_as_imgs(x,y,preds,save_path:str):
    """
    x : input_image of shape (N, C, H, W)
    y : targets of shape (N, 1, H, W)
    pred : model output of shape (N, 1, H, W)
    """
    x = x.cpu()
    y = y.cpu()
    preds = preds.cpu()

    batch_size = x.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(9, 3*batch_size))

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        img = tf.to_pil_image(x[i])
        mask = tf.to_pil_image(y[i].squeeze(0))
        pred = tf.to_pil_image(preds[i].squeeze(0))
    
        axes[i][0].imshow(img)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred, cmap="gray")
        axes[i][2].set_title("Prediction")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

