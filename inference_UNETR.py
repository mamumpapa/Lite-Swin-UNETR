import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from monai.data import CacheDataset, DataLoader

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    Resized,
    NormalizeIntensityd,
    ToTensord,
    AsDiscrete,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandAdjustContrastd,
)
from monai.networks.utils import one_hot
from utils.SegFormer import SegFormer3D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (128, 128, 128)
IN_CHANNEL = 2 #Hippocampus일 경우 1
NUM_CLASS = 3
test_ratio, val_ratio = 0.1, 0.2
RANDOM_SEED = 830

WEIGHTS_PATH = './output/Segmentation(3D)-UNETR(monai)/196_0.304888UNETR.pth'

data_dir = './data/Task05_Prostate'

transforms_val = Compose([
    LoadImaged(keys=("image", "label"), image_only=False),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode='trilinear'),
    Resized(keys=["label"], spatial_size=IMAGE_SIZE, mode='nearest-exact'),
    ToTensord(keys=["image", "label"]),
])

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} 
              for image_name, label_name in zip(train_images, train_labels)]
TrainSet, TestSet = model_selection.train_test_split(data_dicts, test_size=test_ratio, random_state=RANDOM_SEED)
test_ds = CacheDataset(data=TestSet, transform=transforms_val, cache_num=2, cache_rate=1.0, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
OUTPUT_DIR = './output/inference_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

from monai.networks.nets import UNETR

model = UNETR(
    in_channels=IN_CHANNEL,
    out_channels=NUM_CLASS,
    img_size=IMAGE_SIZE,
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).float()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

pred_dict = {'input': [], 'target': [], 'output': []}
with torch.no_grad():
    for i, data in enumerate(test_loader):
        img, target = data["image"], data["label"]
        img_cuda, target_cuda = img.to(DEVICE), target.to(DEVICE)
        output = model(img_cuda)
        output = torch.argmax(output, dim=1).detach().cpu()
        pred_dict['input'].append(img_cuda)
        pred_dict['target'].append(target_cuda)
        pred_dict['output'].append(output)
        if i > 10:
            break

scores = []
fn_ratios = []
fp_ratios = []
class_id = 1


for num in range(1, 5):
    loader_iter = iter(test_loader)
    for i in range(num):
        data = next(loader_iter)
        img, gt = data['image'].to(DEVICE), data['label'].to(DEVICE)
        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1)
            gt_np = gt[0,0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()
            D = gt_np.shape[2]
            for z in range(D):
                gt_mask = (gt_np[:,:,z] == class_id)
                pred_mask = (pred_np[:,:,z] == class_id)
                tp = np.logical_and(gt_mask, pred_mask).sum()
                fp = np.logical_and(~gt_mask, pred_mask).sum()
                fn = np.logical_and(gt_mask, ~pred_mask).sum()
                denom = tp + fp + fn
                ratio = fn / denom if denom > 0 else np.nan
                ratio2 = fp / denom if denom > 0 else np.nan
                fn_ratios.append(ratio)
                fp_ratios.append(ratio2)
                scores.append((z, fn_ratios, fp_ratios))
                if i==num-1:
                    print(f"Slice {z}: FN/(TP+FP+FN) = {ratio:.4f}, FP/(TP+FP+FN) = {ratio2:.4f}")

    output_dir = './output/inference_results/Prostate_UNETR'
    os.makedirs(output_dir, exist_ok=True)


    label_colors = {
        0: [0, 0, 0, 0],
        1: [255, 0, 0, 255],
        2: [0, 255, 0, 255],
    }

    original_3d = pred_dict['input'][0][0, 0]
    target_3d    = pred_dict['target'][0][0, 0]
    output_3d    = pred_dict['output'][0][0]
    nrows, ncols = 5, 10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.5, nrows*1.5))
    interval = max(1, D // (nrows * ncols))
    class_id = 2

    tp_color = [0, 255,   0, 255]
    fp_color = [255, 0,   0, 255]
    fn_color = [0,   0, 255, 255]

    gt_np   = gt[0,0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()
    D = gt_np.shape[2]

    for idx in range(nrows * ncols):
        z = idx * interval
        if z >= D:
            break
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        img_slice = img[0,0,:,:,z].cpu().numpy()
        gt_mask = (gt_np[:,:,z] == class_id)
        pred_mask = (pred_np[:,:,z] == class_id)

        tp = gt_mask & pred_mask
        fp = ~gt_mask & pred_mask
        fn = gt_mask & ~pred_mask

        overlay = np.zeros((*tp.shape, 4), dtype=np.uint8)
        overlay[tp] = [0,255,0,255]
        overlay[fp] = [255,0,0,255]
        overlay[fn] = [0,0,255,255]

        ax.imshow(img_slice, cmap='gray')
        ax.imshow(overlay, alpha=0.6)
        ax.set_title(f"Z={z}", fontsize=6)
        ax.axis('off')

    plt.tight_layout()



    TXT_SAVE_NAME2 = f'{num}_UNETR_Prostate_overlay_result.png'

    save_path = os.path.join(output_dir, TXT_SAVE_NAME2)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Overlay 결과가 '{save_path}'에 저장되었습니다.")
