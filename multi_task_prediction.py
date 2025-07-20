# Imports
import os
import pandas as pd
import glob

from tqdm import tqdm
from monai.networks.nets import resnet
from model_utils import PairedMedicalDataset_Multi_Task, Siamese_Multi_Task_Network

from monai.transforms import (
    Resize,
    ScaleIntensityRange,
    Transpose,
    EnsureType
)

import torch

# Optimize for performance with torch.compile
torch.set_float32_matmul_precision('high')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and names
scans_dir = None
segmentations_dir = None
targets_data_dir = None
output_path = None
model_path = None
model_name = None                                                                 

pd_targets = pd.read_csv(os.path.join(targets_data_dir, "test_data_AmCORE.csv"))

#extract events and targets
path_labels = torch.tensor(pd_targets["path_resp"].values)
pfs_events = torch.tensor(pd_targets["pfsstat"].values)
pfs_targets = torch.tensor(pd_targets["pfs"].values)
os_events = torch.tensor(pd_targets["OSSTAT"].values)
os_targets = torch.tensor(pd_targets["OS"].values)

nifti_images = sorted(glob.glob(os.path.join(scans_dir, "*.nii.gz")))   
npy_segmentations = sorted(glob.glob(os.path.join(segmentations_dir, "*.npy")))

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]
segmentation_pairs = [(npy_segmentations[i], npy_segmentations[i + 1]) for i in range(0, len(npy_segmentations) - 1, 2)]

# Create training and validation datasets
dataset = PairedMedicalDataset_Multi_Task(
    image_pairs, segmentation_pairs, path_labels, pfs_targets, pfs_events, os_targets, os_events, transform=[ScaleIntensityRange(a_min=-100,
                                                                        a_max=200, b_min=0.0, b_max=1.0, clip=True), 
                                                                        Resize((256, 256, 64), 
                                                                        mode="trilinear"),
                                                                        Transpose((0, 3, 2, 1)),
                                                                        EnsureType(data_type="tensor")]
    )

train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

# Init model
encoder = resnet.resnet10(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="B", bias_downsample=False)

model = Siamese_Multi_Task_Network(base_model=encoder, tumor_attn_rate=1, dropout_rate=0.3)
model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
model.to(device)

# Make predictions
model.eval()

path_probs = []
path_preds = []
pfs_preds = []
os_preds = []

for img_1, img_2, attention_mask1, attention_mask2, _, pfs_target, _, _, _ in tqdm(train_loader):
    img_1, img_2, attention_mask1, attention_mask2 = img_1.to(device), img_2.to(device), attention_mask1.to(device), attention_mask2.to(device)
    
    print(pfs_target)
    # Forward pass
    path_resp, log_hz_pfs, log_hz_os, _, _, _, _, _, _ = model(img_1, img_2, attention_mask1, attention_mask2)  # outputs: (batch_size, 1)

    # Apply sigmoid activation to path response
    path_resp_prob = torch.sigmoid(path_resp)
    path_resp_pred = (path_resp_prob > 0.5).int()

    # All predictions are saved, values are masked during metric calculation
    path_probs.extend(path_resp_prob.detach().cpu().numpy().flatten())
    path_preds.extend(path_resp_pred.detach().cpu().numpy().flatten())

    pfs_preds.extend(log_hz_pfs.detach().cpu().numpy().flatten())
    os_preds.extend(log_hz_os.detach().cpu().numpy().flatten())

    torch.cuda.empty_cache()

# Save predictions
results_df = pd.DataFrame({
    "path_probs": path_probs,
    "path_preds": path_preds,
    "pfs_preds": pfs_preds,
    "os_preds": os_preds
})
results_df.to_csv(os.path.join(output_path, "multi_task_model_predictions_training.csv"), index=False)
