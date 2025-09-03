import torch
import torchvision as tv
import torch.nn as nn
from torch.nn import functional as F
import time
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torchsummary import summary as model_summary
import glob
import nibabel as nib
import sklearn
from sklearn import model_selection
import monai
from monai.networks.utils import one_hot
from torchsummary import summary
from torch.cuda.amp import GradScaler, autocast

from monai.networks.nets import UNet
from monai.networks.layers import Norm
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
from monai.config import print_config
from typing import Optional
from tqdm import tqdm
import torch.cuda.amp as amp

import os

RANDOM_SEED = 830
IMAGE_SIZE = (128, 128, 128)
BATCH_SIZE = 1
IN_CHANNEL = 2
NUM_CLASS = 3
EPOCHS = 2
test_ratio, val_ratio = 0.1, 0.2

MODEL_SAVE = True
if MODEL_SAVE:
    model_dir1 = './output/'
    model_dir2 = 'Segmentation(3D)-UNETR(monai)'
    MODEL_SAVE_PATH = os.path.join(model_dir1, model_dir2)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_MY_DATA = True
data_dir = './data/Task05_Prostate'

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} 
              for image_name, label_name in zip(train_images, train_labels)]
TrainSet, TestSet = model_selection.train_test_split(data_dicts, test_size=test_ratio, random_state=RANDOM_SEED)
TrainSet, ValSet = model_selection.train_test_split(TrainSet, test_size=val_ratio, random_state=RANDOM_SEED)
print('TrainSet:', len(TrainSet), 'ValSet:', len(ValSet), 'TestSet:', len(TestSet))
from monai.transforms.compose import MapTransform

class MinMax(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key] - np.min(d[key])
            d[key] = d[key] / np.max(d[key])
        return d

transforms = Compose([
    LoadImaged(keys=("image", "label"), image_only=False),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode='trilinear'),
    Resized(keys=["label"], spatial_size=IMAGE_SIZE, mode='nearest-exact'),
    RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.8, 1.2)),
    ToTensord(keys=["image", "label"]),
])

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

from monai.data import CacheDataset, DataLoader
train_ds = CacheDataset(data=TrainSet, transform=transforms, cache_num=4, cache_rate=1.0, num_workers=0)
val_ds = CacheDataset(data=ValSet, transform=transforms_val, cache_num=2, cache_rate=1.0, num_workers=0)
test_ds = CacheDataset(data=TestSet, transform=transforms_val, cache_num=2, cache_rate=1.0, num_workers=0)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


from utils.efficientunet.efficientunet import get_efficientunet3d_b0


from torchsummary import summary
from thop import profile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Img_size=128
in_channels=2
data = torch.randn((1, in_channels, Img_size, Img_size, Img_size), device=device)
model = get_efficientunet3d_b0(
    out_channels=NUM_CLASS,
    concat_input=True,
    pretrained=False
)
model = model.to(device)
print(model(data).size())
summary(model, input_size=(in_channels, Img_size, Img_size, Img_size))
flops, params = profile(model, inputs=(data,))
print(f"FLOPs: {flops}, Parameters: {params}")


model = model.to(DEVICE)

from monai.losses import DiceCELoss, DiceLoss
torch.backends.cudnn.benchmark = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
LossFuncion = monai.losses.DiceLoss(include_background=True, to_onehot_y=False, softmax=True)

MetricDice = monai.metrics.DiceMetric(include_background=True, reduction="mean")
MetricDicePZ = monai.metrics.DiceMetric(include_background=True, reduction="mean")
MetricDiceTZ = monai.metrics.DiceMetric(include_background=True, reduction="mean")

def BinaryOutput(output, keepdim=True):
    shape = output.shape
    argmax_idx = torch.argmax(output, axis=1, keepdim=True)
    argmax_oh = F.one_hot(argmax_idx, num_classes=NUM_CLASS)
    if keepdim:
        argmax_oh = torch.squeeze(argmax_oh, dim=1)
    if len(shape) == 5:
        argmax_oh = argmax_oh.permute(0,4,1,2,3)
    elif len(shape) == 4:
        argmax_oh = argmax_oh.permute(0,3,1,2)
    return argmax_oh

scaler = GradScaler()

def train(epoch, train_loader):
    model.train()
    mean_epoch_loss = 0
    mean_dice_score = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X EPOCHS)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        y = torch.squeeze(y, dim=1)
        y = one_hot(y[:, None, ...], num_classes=NUM_CLASS)
        with autocast():
            logit_map = model(x)
            loss = LossFuncion(logit_map, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        mean_epoch_loss += loss.item()
        bi_output = BinaryOutput(logit_map)
        MetricDice(bi_output, y)
        dice_score = MetricDice.aggregate().item()
        mean_dice_score += dice_score
        epoch_iterator.set_description(f"Training (Epoch {epoch}/{EPOCHS}) (loss={loss.item():.5f}) (dice={dice_score:.5f})")
    mean_epoch_loss /= len(epoch_iterator)
    mean_dice_score /= len(epoch_iterator)
    return mean_epoch_loss, mean_dice_score

def evaluate(epoch, test_loader):
    model.eval() 
    mean_epoch_loss = 0
    mean_dice_score = 0
    mean_PZ = 0
    mean_TZ = 0
    epoch_iterator = tqdm(test_loader, desc="Evaluating (X / X EPOCHS)", dynamic_ncols=True)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            y = torch.squeeze(y, dim=1)
            y = one_hot(y[:, None, ...], num_classes=NUM_CLASS)
            with torch.amp.autocast('cuda'):
                logit_map = model(x)
                loss = LossFuncion(logit_map, y)
            mean_epoch_loss += loss.item()
            
            bi_output = BinaryOutput(logit_map)
            MetricDice(bi_output, y)
            dice_score = MetricDice.aggregate().item()
            mean_dice_score += dice_score

            MetricDicePZ(bi_output[:, 1:2], y[:, 1:2])
            dice_PZ = MetricDicePZ.aggregate().item()
            MetricDiceTZ(bi_output[:, 2:3], y[:, 2:3])
            dice_TZ = MetricDiceTZ.aggregate().item()
            
            mean_PZ += dice_PZ
            mean_TZ += dice_TZ
            
    n = len(epoch_iterator)
    mean_epoch_loss /= n
    mean_dice_score /= n
    mean_PZ /= n
    mean_TZ /= n
    
    MetricDice.reset()
    MetricDicePZ.reset()
    MetricDiceTZ.reset()
    return mean_epoch_loss, mean_dice_score, mean_PZ, mean_TZ 

start_time = time.time()

losses = {'train': [], 'val': []}
dice_scores = {'train': [], 'val': []}
PZ_scores = {'val': []}
TZ_scores = {'val': []}

best_metric = 999
best_epoch = -1
best_overall = None
best_PZ = None
best_TZ = None

for epoch in range(1, EPOCHS+1):
    train_loss, train_dice = train(epoch, train_loader)
    val_loss, val_dice, val_PZ, val_TZ = evaluate(epoch, val_loader)
    losses['train'].append(train_loss)
    losses['val'].append(val_loss)
    dice_scores['train'].append(train_dice)
    dice_scores['val'].append(val_dice)
    PZ_scores['val'].append(val_PZ)
    TZ_scores['val'].append(val_TZ)
    
    print(f"Epoch {epoch}: Val Loss: {val_loss:.6f}, Overall Dice: {val_dice:.6f}, PZ: {val_PZ:.6f}, TZ: {val_TZ:.6f}")
    
    if val_loss < best_metric:
        best_metric = val_loss
        best_epoch = epoch
        best_overall = val_dice
        best_PZ = val_PZ
        best_TZ = val_TZ
        print(f'Best record! [{epoch}] Val Loss: {val_loss:.6f}, Overall Dice: {val_dice:.6f}')
        if MODEL_SAVE and epoch > (EPOCHS-100):
            model_name = f'{best_epoch}_{best_metric:.6f}EfficientNet.pth'
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_name))
            print('saved model')

mean_loss = sum(losses['val']) / len(losses['val'])
mean_overall = sum(dice_scores['val']) / len(dice_scores['val'])
mean_PZ = sum(PZ_scores['val']) / len(PZ_scores['val'])
mean_TZ = sum(TZ_scores['val']) / len(TZ_scores['val'])

end_time = time.time()
total_training_time = end_time - start_time

TXT_SAVE_PATH = './output/train_result/'
os.makedirs(TXT_SAVE_PATH, exist_ok=True)
TXT_SAVE_NAME = f'epoch{EPOCHS}_Prostate_EfficientUNet.txt'

result_file_path = os.path.join(TXT_SAVE_PATH, TXT_SAVE_NAME)
with open(result_file_path, 'w') as result_file:
    result_file.write(f'Best Loss: {best_metric:.6f}\n')
    result_file.write(f'Best Overall Dice Score: {best_overall:.6f}\n')
    result_file.write(f'Best PZ Dice Score: {best_PZ:.6f}\n')
    result_file.write(f'Best TZ Dice Score: {best_TZ:.6f}\n')
    result_file.write(f'Mean Loss: {mean_loss:.6f}\n')
    result_file.write(f'Mean Overall Dice Score: {mean_overall:.6f}\n')
    result_file.write(f'Mean PZ Dice Score: {mean_PZ:.6f}\n')
    result_file.write(f'Mean TZ Dice Score: {mean_TZ:.6f}\n')
    result_file.write(f'Total Training Time: {total_training_time:.2f} seconds\n')
print('Total Training Time: ', total_training_time)

epochs = list(range(len(losses['train'])))
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].plot(epochs, losses['train'], 'g-o', label='Training Loss')
ax[0].plot(epochs, losses['val'], 'r-o', label='Validation Loss')
ax[0].set_title('Training & Validation Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[1].plot(epochs, dice_scores['train'], 'go-', label='Training Overall Dice')
ax[1].plot(epochs, dice_scores['val'], 'ro-', label='Validation Overall Dice')
ax[1].set_title('Training & Validation Overall Dice')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Dice Score")
file_name = f'./output/result_images/{TXT_SAVE_NAME}.png'
plt.savefig(file_name)
plt.show()

pred_dict = {'input': [], 'target': [], 'output': []}
if MODEL_SAVE:
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, model_name)))
model.eval()
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

output_dir = './output/output_images'
os.makedirs(output_dir, exist_ok=True)


label_colors = {
    0: [0, 0, 0, 0],
    1: [255, 0, 0, 255],
    2: [0, 255, 0, 255],
}

original_3d = pred_dict['input'][0][0, 0]
target_3d    = pred_dict['target'][0][0, 0]
output_3d    = pred_dict['output'][0][0]
n_slices = 10
nrows, ncols = 1, n_slices
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))

interval = max(1, IMAGE_SIZE[-1] // n_slices)

class_id = 2

tp_color = [0, 255,   0, 255]
fp_color = [255, 0,   0, 255]
fn_color = [0,   0, 255, 255]

gt_np   = target_3d.cpu().numpy()
pred_np = output_3d.cpu().numpy()

for col in range(ncols):
    slice_idx = col * interval
    if slice_idx >= IMAGE_SIZE[-1]:
        break

    img = original_3d[:, :, slice_idx].cpu().numpy()

    gt_mask   = (gt_np[:, :, slice_idx]   == class_id)
    pred_mask = (pred_np[:, :, slice_idx] == class_id)

    tp = gt_mask & pred_mask
    fp = ~gt_mask & pred_mask
    fn = gt_mask & ~pred_mask

    h, w = tp.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[tp] = tp_color
    overlay[fp] = fp_color
    overlay[fn] = fn_color

    ax = axes[col] if ncols>1 else axes
    ax.imshow(img, cmap='gray')
    ax.imshow(overlay, alpha=0.6)
    ax.set_title(f"Slice {slice_idx}\nTP/FP/FN", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()

TXT_SAVE_NAME2 = f'epoch{EPOCHS}_Prostate_EfficientUNet_overlay_result.png'
save_path = os.path.join(output_dir, TXT_SAVE_NAME2)
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Overlay 결과가 '{save_path}'에 저장되었습니다.")