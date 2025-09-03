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
IN_CHANNEL = 2#Hippocampus일 경우 1
NUM_CLASS = 3
test_ratio, val_ratio = 0.1, 0.2
RANDOM_SEED = 830

WEIGHTS_PATH = './output/Segmentation(3D)-UNETR(monai)/113_0.273156Lite_Swin_UNETR.pth'

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
OUTPUT_DIR = './output/inference_results_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

from utils.advanced_swin_unetr import SwinUNETR


model = SwinUNETR(
    img_size=IMAGE_SIZE,
    in_channels=IN_CHANNEL,
    out_channels=NUM_CLASS,
    feature_size=12,
    use_checkpoint=False,
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
        if i > 26:
            break

scores = []
fn_ratios = []
fp_ratios = []
class_id = 1

for num in range(1, 27):
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

    output_dir = './output/inference_results_v2/Prostate_Lite_Swin_UNETR'  # 경로 수정
    os.makedirs(output_dir, exist_ok=True)


    offsets   = [0, 20, 40, 60]
    interval  = 2
    n_blocks  = len(offsets)
    nrows     = n_blocks * 2
    ncols     = 10

    label_colors = {
        0: [  0,   0,   0,   0],
        1: [255,   0,   0, 255],
        2: [  0, 255,   0, 255],
    }

    def create_overlay(label_slice):
        h, w = label_slice.shape
        ov = np.zeros((h, w, 4), dtype=np.uint8)
        for lbl, rgba in label_colors.items():
            ov[label_slice == lbl] = rgba
        return ov
    sample_idx = num - 1

    orig_vol = pred_dict['input'][sample_idx][0, 0].cpu().numpy()
    gt_vol   = pred_dict['target'][sample_idx][0, 0].cpu().numpy().astype(np.uint8)
    pr_vol   = pred_dict['output'][sample_idx][0].cpu().numpy().astype(np.uint8)
    _, _, D  = orig_vol.shape

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(ncols*1.2, nrows*1.2))

    for row in range(nrows):
        block_idx = row // 2
        is_pred   = (row % 2) == 1
        offset    = offsets[block_idx]

        for col in range(ncols):
            z = offset + col * interval
            ax = axes[row, col]

            if z >= D:
                ax.axis('off')
                continue

            ax.imshow(orig_vol[:, :, z], cmap='gray')

            mask = pr_vol[:, :, z] if is_pred else gt_vol[:, :, z]
            ov   = create_overlay(mask)
            ax.imshow(ov, alpha=0.4)

            kind = "PR" if is_pred else "GT"
            ax.set_title(f"{kind} Z={z}", fontsize=6)
            ax.axis('off')
    plt.tight_layout()


    

    TXT_SAVE_NAME2 = f'{num}_Lite_Swin_UNETR_Prostate_overlay_result.png'

    save_path = os.path.join(output_dir, TXT_SAVE_NAME2)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Overlay 결과가 '{save_path}'에 저장되었습니다.")