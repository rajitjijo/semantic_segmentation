import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms

class CarvanaDataset(Dataset):

    def __init__(self, image_dir, masks_dir, transforms=None):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.masks_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)

