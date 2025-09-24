import os, shutil, random

img_dir = "data/train/train"
masks_dir = "data/train_masks/train_masks"
val_img_dir = "data/val"
val_mask_dir = "data/val_masks"

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

all_images = sorted(os.listdir(img_dir))
random.shuffle(all_images)

val_size = int(0.2 * len(all_images))
val_images = all_images[:val_size]

#'fff9b3a5373f_12_mask.gif'
#'843763f47895_06.jpg'

for fname in val_images:

    mask = fname.replace(".jpg", "_mask.gif")
    
    img_src = os.path.join(img_dir, fname)
    mask_src = os.path.join(masks_dir, mask)
    img_dst = os.path.join(val_img_dir, fname)
    mask_dst = os.path.join(val_mask_dir, mask)

    if os.path.exists(mask_src):
        shutil.move(img_src, img_dst)
        shutil.move(mask_src, mask_dst)
        print(f"Moving {img_src} to {img_dst}")
    else:
        print(f"Couldnt find mask for {img_src}")

