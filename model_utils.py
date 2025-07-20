# Imports
import numpy as np
import nibabel as nib
from monai.data import Dataset
from monai.transforms import (
    Compose,
    Transpose
)
import torch
import torch.nn as nn
from torchsurv.loss.cox import neg_partial_log_likelihood


# ------------------------------------------------------
# DATASETS
# ------------------------------------------------------
    
    
class PairedMedicalDataset_Multi_Task(Dataset):
    def __init__(self, image_pairs, attention_pairs, path_labels, pfs_targets, pfs_events, os_targets, os_events, transform=None):
        self.image_pairs = image_pairs
        self.attention_pairs = attention_pairs
        self.path_labels = path_labels
        self.pfs_targets = pfs_targets
        self.pfs_events = pfs_events
        self.os_targets = os_targets
        self.os_events = os_events
        self.transform = Compose(transform)
        self.transform_attention = Compose([Transpose((0, 3, 2, 1))])

    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        att1_path, att2_path = self.attention_pairs[idx]

        # Load images using nibabel (for NIfTI)
        img1 = nib.load(img1_path).get_fdata()
        img2 = nib.load(img2_path).get_fdata()

        # Add channel dimension for CNN input (C, H, W, D)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        # Load attention masks
        att1 = np.load(att1_path)
        att2 = np.load(att2_path)

        # Add channel dimension for CNN input (C, H, W, D)
        att1 = np.expand_dims(att1, axis=0)
        att2 = np.expand_dims(att2, axis=0)

        att1 = self.transform_attention(att1)
        att2 = self.transform_attention(att2)

        path_labels = self.path_labels[idx].float()
        pfs_target = self.pfs_targets[idx].float()
        pfs_event = self.pfs_events[idx]
        os_target = self.os_targets[idx].float()
        os_event = self.os_events[idx]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1.float().as_tensor(), img2.float().as_tensor(), att1.float().as_tensor(), att2.float().as_tensor(), path_labels, pfs_target, pfs_event, os_target, os_event


# ------------------------------------------------------
# NETWORKS
# ------------------------------------------------------

class Siamese_Multi_Task_Network(nn.Module):
    def __init__(self, base_model, tumor_attn_rate=0.7, dropout_rate=0.4):
        super(Siamese_Multi_Task_Network, self).__init__()
        """
        Initializes the Siamese Multi-Task Network with a shared encoder and multiple task-specific classifiers.
        Uses deep supervision and lesion aware attention to focus on tumor regions.
        """

        # How much to focus on tumor regions
        self.tumor_attn_rate = tumor_attn_rate

        self.encoder_stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.act,
            base_model.maxpool,
            base_model.layer1,
        )
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifiers for different tasks
        self.path_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256,1)
        )
        
        self.pfs_regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256,1)
        )
        
        self.os_regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256,1)
        )

        self.aux_dropout = nn.Dropout(dropout_rate)

        # Deep supervision classifiers
        self.aux_clf2_path = nn.Linear(256, 1)
        self.aux_clf3_path = nn.Linear(512, 1)
        self.aux_rgr2_pfs = nn.Linear(256, 1)
        self.aux_rgr3_pfs = nn.Linear(512, 1)
        self.aux_rgr2_os = nn.Linear(256, 1)
        self.aux_rgr3_os = nn.Linear(512, 1)


    def forward(self, image1, image2, attention_mask1, attention_mask2):
        # Pass both inputs through the shared model
        output1 = self.encoder_stem(image1)
        output2 = self.encoder_stem(image2)

        # Apply attention masks (output shape layer1: (16, 64, 64))
        soft_mask1 = (attention_mask1 == 1) * self.tumor_attn_rate + (attention_mask1 == 0) * (1 - self.tumor_attn_rate)
        soft_mask2 = (attention_mask2 == 1) * self.tumor_attn_rate + (attention_mask2 == 0) * (1 - self.tumor_attn_rate)

        output1 = output1 * soft_mask1
        output2 = output2 * soft_mask2  

        # Layer 2
        output1 = self.layer2(output1)
        output2 = self.layer2(output2)
        a2_1 = self.adaptive_pool(output1).view(output1.size(0), -1)
        a2_2 = self.adaptive_pool(output2).view(output2.size(0), -1)
        combined_aux_2 = torch.cat([a2_1, a2_2], dim=1)
        combined_aux_2 = self.aux_dropout(combined_aux_2)
        aux_out2_path = self.aux_clf2_path(combined_aux_2)
        aux_out2_pfs = self.aux_rgr2_pfs(combined_aux_2)
        aux_out2_os = self.aux_rgr2_os(combined_aux_2)

        # Layer 3
        output1 = self.layer3(output1)
        output2 = self.layer3(output2)
        a3_1 = self.adaptive_pool(output1).view(output1.size(0), -1)
        a3_2 = self.adaptive_pool(output2).view(output2.size(0), -1)
        combined_aux_3 = torch.cat([a3_1, a3_2], dim=1)
        combined_aux_3 = self.aux_dropout(combined_aux_3)
        aux_out3_path = self.aux_clf3_path(combined_aux_3)
        aux_out3_pfs = self.aux_rgr3_pfs(combined_aux_3)
        aux_out3_os = self.aux_rgr3_os(combined_aux_3)

        # Layer 4 (final layer)
        output1 = self.layer4(output1)
        output2 = self.layer4(output2)

        # Apply adaptive average pooling to both outputs
        output1 = self.adaptive_pool(output1).view(output1.size(0), -1)
        output2 = self.adaptive_pool(output2).view(output2.size(0), -1)
        combined_embeddings = torch.cat((output1, output2), dim=1)

        # Final outputs for each task
        path_output = self.path_classifier(combined_embeddings)
        pfs_output = self.pfs_regressor(combined_embeddings)
        os_output = self.os_regressor(combined_embeddings)


        return path_output, pfs_output, os_output, aux_out2_path, aux_out3_path, aux_out2_pfs, aux_out3_pfs, aux_out2_os, aux_out3_os
    
# ------------------------------------------------------
# LOSS FUNCTIONS
# ------------------------------------------------------


class MultiTaskDeepSupervisionLoss(nn.Module):
    def __init__(self, aux_weight=0.5, pos_weight=torch.tensor([0.73])):
        super().__init__()
        self.aux_weight = aux_weight
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    def masked_bce_loss(self, pred, target):
        """Binary Cross Entropy with masking (-1 = ignore)"""
        mask = (target != -1)
        if mask.sum() == 0:
            # Return dummy zero loss that allows gradient flow
            return torch.sum(pred * 0.0)
        return self.bce_logits(pred[mask], target[mask].float())


    def forward(self, outputs, targets):
        (
            path_resp, log_hz_pfs, log_hz_os,
            aux2_path, aux3_path,
            aux2_pfs, aux3_pfs,
            aux2_os, aux3_os
        ) = outputs

        (
            label,              # binary class label (with -1 for missing)
            durations_pfs, events_pfs,
            durations_os, events_os
        ) = targets

        # --- Path resp. Loss (Main + Aux), with masking
        loss_path_main = self.masked_bce_loss(path_resp, label)
        loss_path_aux2 = self.masked_bce_loss(aux2_path, label)
        loss_path_aux3 = self.masked_bce_loss(aux3_path, label)
        loss_path = (
            loss_path_main
            + self.aux_weight * (loss_path_aux2 + loss_path_aux3)
        )

        # --- Cox Loss: PFS
        loss_pfs_main = neg_partial_log_likelihood(log_hz_pfs, events_pfs, durations_pfs)
        loss_pfs_aux2 = neg_partial_log_likelihood(aux2_pfs, events_pfs, durations_pfs)
        loss_pfs_aux3 = neg_partial_log_likelihood(aux3_pfs, events_pfs, durations_pfs)
        loss_pfs = (
            loss_pfs_main
            + self.aux_weight * (loss_pfs_aux2 + loss_pfs_aux3)
        )

        # --- Cox Loss: OS
        loss_os_main = neg_partial_log_likelihood(log_hz_os, events_os, durations_os)
        loss_os_aux2 = neg_partial_log_likelihood(aux2_os, events_os, durations_os)
        loss_os_aux3 = neg_partial_log_likelihood(aux3_os, events_os, durations_os)
        loss_os = (
            loss_os_main
            + self.aux_weight * (loss_os_aux2 + loss_os_aux3)
        )

        # --- Total loss (divide by number of events to normalize)
        total_loss = loss_path + (loss_pfs/(events_pfs.sum() + 1e-8)) + (loss_os/(events_os.sum() + 1e-8))  

        return total_loss, {
            "loss_path": loss_path.item(),
            "loss_pfs": loss_pfs.item(),
            "loss_os": loss_os.item(),
        }
