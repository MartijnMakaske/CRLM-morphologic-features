# Imports
import torch
import os
import nibabel as nib

import numpy as np
from monai.transforms import (
    Resize,
    ScaleIntensityRange,
    Transpose,
    Compose
)

from model_utils import Siamese_Multi_Task_Network

from monai.networks.nets import resnet

from captum.attr import IntegratedGradients, NoiseTunnel
from tqdm import tqdm
import torch.nn as nn



class WrappedSiameseModel(nn.Module):
    def __init__(self, siamese_model, attn1, attn2):
        super().__init__()
        self.model = siamese_model
        self.attn1 = attn1.to(device)
        self.attn2 = attn2.to(device)

    def forward(self, input_combined):
        # input_combined shape: [1, 2, D, H, W]
        img1 = input_combined[:, 0:1, ...]
        img2 = input_combined[:, 1:2, ...]
        path_resp, log_hz_pfs, log_hz_os = self.model(img1, img2, self.attn1, self.attn2)[0:3]
        return torch.cat([path_resp, log_hz_pfs, log_hz_os], dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#encoder18 = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True)
encoder10 = resnet.resnet10(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="B", bias_downsample=False)

model = Siamese_Multi_Task_Network(encoder10, tumor_attn_rate=1)
model.load_state_dict(torch.load("./models/Multi_Task_lr3_100epochs_batch_acc64_patience15_resnet10_dropout3_xtra_data.pth"))

# Paths
test_scans_path = None
test_segm_path = None
save_path = None


transform_attention = Compose([Transpose((0, 3, 2, 1))])

transform = [
    ScaleIntensityRange(a_min=-100, a_max=200, b_min=0.0, b_max=1.0, clip=True),
    Resize((256, 256, 64), mode="trilinear"),
    Transpose((0, 3, 2, 1))
]

# Iterate through the DataLoader
scans = [scan for scan in os.listdir(test_scans_path) if scan.split("_")[1][0] == "0"]

for scan in tqdm(scans, desc="Processing scans"):
    print(f"Processing patient ID: {scan}")
    patient_id = scan.split("_")[0]

    # Load Images and Segmentations
    image1_path = os.path.join(test_scans_path, scan)
    image2_path = os.path.join(test_scans_path, scan.replace("_0", "_1"))
    segm1_path = os.path.join(test_segm_path, scan.replace(".nii.gz", ".npy"))
    segm2_path = os.path.join(test_segm_path, scan.replace("_0", "_1").replace(".nii.gz", ".npy"))

    img1_nifti = nib.load(image1_path)
    img2_nifti = nib.load(image2_path)
    segm1 = np.load(segm1_path)
    segm2 = np.load(segm2_path)

    img1 = img1_nifti.get_fdata()
    img2 = img2_nifti.get_fdata()

    # Add channel dimension to image data
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    segm1 = np.expand_dims(segm1, axis=0)
    segm2 = np.expand_dims(segm2, axis=0)

    # Apply transformations to CT scans
    img1 = Compose(transform)(img1)
    img2 = Compose(transform)(img2)

    # Apply transformations to segmentations
    segm1 = Compose(transform_attention)(segm1)
    segm2 = Compose(transform_attention)(segm2)

    # Convert and move images to the device
    img1 = img1.float().as_tensor()
    img2 = img2.float().as_tensor()
    segm1 = segm1.float().as_tensor()
    segm2 = segm2.float().as_tensor()

    img1 = torch.unsqueeze(img1, 0)
    img2 = torch.unsqueeze(img2, 0)
    segm1 = torch.unsqueeze(segm1, 0)
    segm2 = torch.unsqueeze(segm2, 0)

    # Combine images for the model
    input_combined = torch.cat([img1, img2], dim=1).requires_grad_() 
    input_combined = input_combined.to(device)

    wrapped_model = WrappedSiameseModel(model, segm1, segm2).to(device).eval()

    baselines = torch.zeros_like(input_combined)

    # Run IG
    ig = IntegratedGradients(wrapped_model)
    nt = NoiseTunnel(ig)

    baseline = torch.zeros_like(input_combined).to(device)

    attributions = nt.attribute(input_combined,
                            baselines=baselines,
                            nt_type='smoothgrad', 
                            stdevs=0.1,
                            nt_samples=20,                   #set back to 10 for faster compute 
                            nt_samples_batch_size=2, 
                            target= 0,  # Path response
                            n_steps= 50,
                            internal_batch_size=4)          

    # For the first input image
    img1_attr = attributions[0, 0].detach().cpu()  # [D, H, W]

    # For the second input image
    img2_attr = attributions[0, 1].detach().cpu()

    # Normalize the saliency map for visualization (should not be performed when showing niftis in 3d slicer)
    img1_attr_normalized = (img1_attr - img1_attr.min()) / (img1_attr.max() - img1_attr.min())
    img2_attr_normalized = (img2_attr - img2_attr.min()) / (img2_attr.max() - img2_attr.min())

    # Create new nifti images for the attributions
    img1_attr_normalized = img1_attr_normalized.numpy()
    img2_attr_normalized = img2_attr_normalized.numpy()

    img1_attr_normalized = np.transpose(img1_attr_normalized, (2,1,0))
    img2_attr_normalized = np.transpose(img2_attr_normalized, (2,1,0))

    nifti_img1_attr = nib.Nifti1Image(img1_attr_normalized, affine=img1_nifti.affine, header=img1_nifti.header)
    nifti_img2_attr = nib.Nifti1Image(img2_attr_normalized, affine=img2_nifti.affine, header=img2_nifti.header)

    
    # Save the attributions
    img1_attr_path = os.path.join(save_path, f"{patient_id}_0_attr_PR.nii.gz")
    img2_attr_path = os.path.join(save_path, f"{patient_id}_1_attr_PR.nii.gz")
    nib.save(nifti_img1_attr, img1_attr_path)
    nib.save(nifti_img2_attr, img2_attr_path)
    print(f"Saved attributions for patient {patient_id}")

        
