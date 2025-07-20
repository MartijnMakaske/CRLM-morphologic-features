#imports
import os
import pandas as pd
import glob

from tqdm import tqdm
from monai.networks.nets import resnet
from model_utils import PairedMedicalDataset_Multi_Task, Siamese_Multi_Task_Network, MultiTaskDeepSupervisionLoss

from monai.transforms import (
    Resize,
    ScaleIntensityRange,
    Transpose,
    EnsureType
)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import wandb

from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex

# Optimize for performance with torch.compile
torch.set_float32_matmul_precision('high')



def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=10, accum_steps=32, device="cuda"):
    global run
    global model_name

    best_val_score = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 20)

        # ---------------------------
        # TRAINING PHASE
        # ---------------------------
        model.train()

        running_loss = 0.0

        # Track for metrics
        all_path_preds = []
        all_path_labels = []

        all_pfs_preds = []
        all_pfs_durations = []
        all_pfs_events = []

        all_os_preds = []
        all_os_durations = []
        all_os_events = []

        optimizer.zero_grad()  

        for step, (train_img_1, train_img_2, attention_mask1, attention_mask2, path_labels, pfs_targets, pfs_events, os_targets, os_events) in enumerate(tqdm(train_loader)):
            train_img_1, train_img_2 = train_img_1.to(device), train_img_2.to(device)
            attention_mask1, attention_mask2 = attention_mask1.to(device), attention_mask2.to(device)
            path_labels = path_labels.to(device)
            pfs_targets, pfs_events = pfs_targets.to(device), pfs_events.to(device)
            os_targets, os_events = os_targets.to(device), os_events.to(device)
            

            # Forward pass
            path_resp, log_hz_pfs, log_hz_os, aux2_path, aux3_path, aux2_pfs, aux3_pfs, aux2_os, aux3_os = model(
                train_img_1, train_img_2, attention_mask1, attention_mask2)

            path_labels = torch.unsqueeze(path_labels, 1)
            log_hz_pfs = torch.squeeze(log_hz_pfs, 1)
            log_hz_os = torch.squeeze(log_hz_os, 1)
            aux2_pfs = torch.squeeze(aux2_pfs, 1)
            aux3_pfs = torch.squeeze(aux3_pfs, 1)
            aux2_os = torch.squeeze(aux2_os, 1)
            aux3_os = torch.squeeze(aux3_os, 1)

            # Compute loss
            loss, loss_dict = criterion(
                outputs=(path_resp, log_hz_pfs, log_hz_os, aux2_path, aux3_path, aux2_pfs, aux3_pfs, aux2_os, aux3_os),
                targets=(path_labels, pfs_targets, pfs_events, os_targets, os_events)
            )

            # Normalize loss by accumulation steps
            loss = loss / accum_steps
            loss.backward()

            # Step optimizer every `accum_steps`
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Store classification predictions
            path_resp = torch.sigmoid(path_resp)
            mask = (path_labels != -1)
            if mask.sum() > 0:
                all_path_preds.append(path_resp[mask].detach().cpu())
                all_path_labels.append(path_labels[mask].detach().cpu())

            # Store survival outputs
            all_pfs_preds.append(log_hz_pfs.detach().cpu())
            all_pfs_durations.append(pfs_targets.detach().cpu())
            all_pfs_events.append(pfs_events.detach().cpu())

            all_os_preds.append(log_hz_os.detach().cpu())
            all_os_durations.append(os_targets.detach().cpu())
            all_os_events.append(os_events.detach().cpu())

            running_loss += loss.item() * accum_steps  # Reverse normalization to track actual loss

            torch.cuda.empty_cache()


        epoch_loss = running_loss / len(train_loader)

        # Stack all predictions and labels
        if all_path_labels:
            path_preds = torch.cat(all_path_preds).squeeze()
            path_labels_cat = torch.cat(all_path_labels).squeeze()
            acc = ((path_preds > 0.5).int() == path_labels_cat.int()).float().mean().item()
        else:
            print("Classification Accuracy: N/A (no valid labels this epoch)")

        path_preds = path_preds.numpy()
        path_labels_cat = path_labels_cat.numpy()

        path_auc = roc_auc_score(path_labels_cat, path_preds)


        # --- Cox Concordance Index (PFS)
        pfs_risk = torch.cat(all_pfs_preds).squeeze()
        pfs_durations = torch.cat(all_pfs_durations).squeeze()
        pfs_events = torch.cat(all_pfs_events).squeeze().bool()

        # --- Cox Concordance Index (OS)
        os_risk = torch.cat(all_os_preds).squeeze()
        os_durations = torch.cat(all_os_durations).squeeze()
        os_events = torch.cat(all_os_events).squeeze().bool()

        # Calculate C-index for PFS and OS
        c_index_pfs = ConcordanceIndex()
        c_index_os = ConcordanceIndex()
        c_index_pfs = c_index_pfs(pfs_risk, pfs_events, pfs_durations)
        c_index_os = c_index_os(os_risk, os_events, os_durations)
        

        # Log metrics to wandb
        run.log({"train loss": epoch_loss, "train Accuracy Path. Resp.": acc, "train path auc": path_auc, "train C-index pfs": c_index_pfs, "train C-index os": c_index_os, "Learning rate": scheduler.get_last_lr()[-1]})
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy Path Resp.: {acc:.4f} | Path. Resp. AUC: {path_auc:.4f} | C-index PFS: {c_index_pfs:.4f} | C-index OS: {c_index_os:.4f} | Learning rate: {scheduler.get_last_lr()[-1]}")
        print(f"Loss path resp: {loss_dict['loss_path']:.4f} | Loss pfs: {loss_dict['loss_pfs']:.4f} | Loss os: {loss_dict['loss_os']:.4f}")

        val_score = validate_model(model, val_loader, criterion, device)

        # Save best model based on validation loss
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), f'./models/{model_name}.pth')
            print("Best model saved!")
            early_stopping_counter = 0  # Reset counter if we improve
        else:
            early_stopping_counter += 1
            
        # Early stopping
        if early_stopping_counter >= 40:  # If no improvement for 40 epochs
            print(f"Early stopping triggered after {early_stopping_counter} epochs without improvement.")
            break

        # Learning rate scheduler step
        scheduler.step(val_score)



def validate_model(model, val_loader, criterion, device="cuda"):
    global run

    model.eval()
    running_loss = 0.0

    # Track for metrics
    all_path_preds = []
    all_path_labels = []

    all_pfs_preds = []
    all_pfs_durations = []
    all_pfs_events = []

    all_os_preds = []
    all_os_durations = []
    all_os_events = []

    with torch.no_grad():
        for val_img_1, val_img_2, attention_mask1, attention_mask2, path_labels, pfs_targets, pfs_events, os_targets, os_events in tqdm(val_loader):
            val_img_1, val_img_2, attention_mask1, attention_mask2, path_labels, pfs_targets, pfs_events, os_targets, os_events = val_img_1.to(device), val_img_2.to(device), attention_mask1.to(device), attention_mask2.to(device), path_labels.to(device), pfs_targets.to(device), pfs_events.to(device), os_targets.to(device), os_events.to(device)

            # Forward pass
            path_resp, log_hz_pfs, log_hz_os, aux2_path, aux3_path, aux2_pfs, aux3_pfs, aux2_os, aux3_os = model(val_img_1, val_img_2, attention_mask1, attention_mask2)

            # Unsqueeze path response labels
            path_labels = torch.unsqueeze(path_labels, 1)
            # Squeeze output tensors
            log_hz_pfs = torch.squeeze(log_hz_pfs, 1)
            log_hz_os = torch.squeeze(log_hz_os, 1)
            aux2_pfs = torch.squeeze(aux2_pfs, 1)
            aux3_pfs = torch.squeeze(aux3_pfs, 1)
            aux2_os = torch.squeeze(aux2_os, 1)
            aux3_os = torch.squeeze(aux3_os, 1)

            # Compute loss
            loss, loss_dict = criterion(outputs=(path_resp, log_hz_pfs, log_hz_os, aux2_path, aux3_path, aux2_pfs, aux3_pfs, aux2_os, aux3_os),
                             targets=(path_labels, pfs_targets, pfs_events, os_targets, os_events),
                             )

            running_loss += loss.item()

            # Apply sigmoid activation to path response
            path_resp = torch.sigmoid(path_resp)

            # Store predictions
            mask = (path_labels != -1)
            if mask.sum() > 0:
                all_path_preds.append(path_resp[mask].detach().cpu())
                all_path_labels.append(path_labels[mask].detach().cpu())

            all_pfs_preds.append(log_hz_pfs.detach().cpu())
            all_pfs_durations.append(pfs_targets.detach().cpu())
            all_pfs_events.append(pfs_events.detach().cpu())

            all_os_preds.append(log_hz_os.detach().cpu())
            all_os_durations.append(os_targets.detach().cpu())
            all_os_events.append(os_events.detach().cpu())

    val_loss = running_loss / len(val_loader)

    # Compute metrics
    if all_path_labels:
        path_preds = torch.cat(all_path_preds).squeeze()
        path_labels_cat = torch.cat(all_path_labels).squeeze()
        acc = ((path_preds > 0.5).int() == path_labels_cat.int()).float().mean().item()
    else:
        print("Validation Accuracy: N/A (no valid labels this epoch)")

    # AUC ROC
    path_preds = path_preds.numpy()
    path_labels_cat = path_labels_cat.numpy()

    path_auc = roc_auc_score(path_labels_cat, path_preds)

    # --- Cox Concordance Index (PFS)
    pfs_risk = torch.cat(all_pfs_preds).squeeze()
    pfs_durations = torch.cat(all_pfs_durations).squeeze()
    pfs_events = torch.cat(all_pfs_events).squeeze().bool()
    # --- Cox Concordance Index (OS)
    os_risk = torch.cat(all_os_preds).squeeze()
    os_durations = torch.cat(all_os_durations).squeeze()
    os_events = torch.cat(all_os_events).squeeze().bool()

    # Compute metrics
    c_index = ConcordanceIndex()
    c_index_pfs = c_index(pfs_risk, pfs_events, pfs_durations)
    c_index_os = c_index(os_risk, os_events, os_durations)

    # Log or print
    print(f"Validation Loss: {val_loss:.4f} | Accuracy Path Resp.: {acc if acc is not None else 'N/A'} | "
          f"Path. Resp. AUC: {path_auc:4f} | "
          f"C-index PFS: {c_index_pfs:.4f} | "
          f"C-index OS: {c_index_os:.4f}")
    
    combined_score = path_auc + c_index_pfs + c_index_os

    run.log({"val loss": val_loss,
             "val Accuracy Path. Resp.": acc,
             "val path AUC": path_auc,
             "val C-index pfs": c_index_pfs, 
             "val C-index os": c_index_os,
             "val score": combined_score})
    
    return combined_score



# ---------------------------------------
# READ DATA
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scans_dir = None
segmentations_dir = None
targets_data_dir = None

nifti_images = sorted(glob.glob(os.path.join(scans_dir, "*.nii.gz")))   
nifti_segmentations = sorted(glob.glob(os.path.join(segmentations_dir, "*.npy")))

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]
segmentation_pairs = [(nifti_segmentations[i], nifti_segmentations[i + 1]) for i in range(0, len(nifti_segmentations) - 1, 2)]

pd_targets = pd.read_csv(os.path.join(targets_data_dir, "training_data.csv"))

#extract events and targets
path_labels = torch.tensor(pd_targets["path_resp"].values)
pfs_events = torch.tensor(pd_targets["pfsstat"].values)
pfs_targets = torch.tensor(pd_targets["pfs"].values)
os_events = torch.tensor(pd_targets["OSSTAT"].values)
os_targets = torch.tensor(pd_targets["OS"].values)


# ---------------------------------------
# HYPERPARAMETERS
# ---------------------------------------
batch_size = 8
accum_steps = 8  # Gradient accumulation steps     (effective batch size of 64)
num_epochs = 100
learning_rate = 1e-3
dropout_rate = 0.3
lr_patience = 20       
tumor_attn_rate = 1
model_name = "Multi_Task_lr3_100epochs_batch_acc64_patience15_resnet10_dropout3_xtra_data"

print(f"model name: {model_name}")

# Perform a single train-test split with stratification
train_image_pairs, val_image_pairs, train_seg_pairs, val_seg_pairs, train_path_labels, val_path_labels, train_pfs_targets, val_pfs_targets, train_pfs_events, val_pfs_events, train_os_targets, val_os_targets, train_os_events, val_os_events = train_test_split(
    image_pairs, segmentation_pairs, path_labels, pfs_targets, pfs_events, os_targets, os_events, 
    test_size=0.2, random_state=42, stratify=path_labels
)

for i in val_image_pairs:
    print(f"Validation image pair: {(i[0].split('/'))[-1]} and {(i[1].split('/'))[-1]}")


# Create training and validation datasets
train_dataset = PairedMedicalDataset_Multi_Task(
    train_image_pairs, train_seg_pairs, train_path_labels, train_pfs_targets, train_pfs_events, train_os_targets, train_os_events, transform=[ScaleIntensityRange(a_min=-100,
                                                                        a_max=200, b_min=0.0, b_max=1.0, clip=True), 
                                                                        Resize((256, 256, 64), 
                                                                        mode="trilinear"),
                                                                        Transpose((0, 3, 2, 1)),
                                                                        EnsureType(data_type="tensor")]
)
val_dataset = PairedMedicalDataset_Multi_Task(
    val_image_pairs, val_seg_pairs, val_path_labels, val_pfs_targets, val_pfs_events, val_os_targets, val_os_events, transform=[ScaleIntensityRange(a_min=-100,
                                                                        a_max=200, b_min=0.0, b_max=1.0, clip=True), 
                                                                        Resize((256, 256, 64), 
                                                                        mode="trilinear"),
                                                                        Transpose((0, 3, 2, 1)),
                                                                        EnsureType(data_type="tensor")]
    )

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize the model
encoder = resnet.resnet10(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="B", bias_downsample=False)


#model = torch.compile(SiameseNetwork_Images_OS(encoder))
model = Siamese_Multi_Task_Network(encoder, tumor_attn_rate, dropout_rate)
model = model.to(device)


regressor_params = list(model.path_classifier.parameters()) + \
                    list(model.aux_clf2_path.parameters()) + \
                    list(model.aux_clf3_path.parameters()) + \
                    list(model.pfs_regressor.parameters()) + \
                    list(model.aux_rgr2_pfs.parameters()) + \
                    list(model.aux_rgr3_pfs.parameters()) + \
                    list(model.os_regressor.parameters()) + \
                    list(model.aux_rgr2_os.parameters()) + \
                    list(model.aux_rgr3_os.parameters())
                    

# Optimizer
optimizer = optim.AdamW([
    {'params': model.encoder_stem.parameters(), 'lr': learning_rate * 0.1},
    {'params': model.layer2.parameters(), 'lr': learning_rate * 0.1},
    {'params': model.layer3.parameters(), 'lr': learning_rate},
    {'params': model.layer4.parameters(), 'lr': learning_rate},
    {'params': regressor_params, 'lr': learning_rate}
], lr=learning_rate, weight_decay=1e-3)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=lr_patience, cooldown=10)

# Loss function, 
pos_weight = torch.tensor([0.73]).to(device)  
criterion = MultiTaskDeepSupervisionLoss(aux_weight=0.3, pos_weight=pos_weight)

#-------------------------------------
# TRACK WITH WANDB
#-------------------------------------
"""
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="martijnmakaske-vrije-universiteit-amsterdam",
    # Set the wandb project where this run will be logged.
    project="CRLM-morph-features",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": learning_rate,
        "architecture": "Siames-Morph-CNN",
        "dataset": "CAIRO5 images",
        "epochs": num_epochs,
    },
)


# Train the model for this fold
train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=num_epochs,accum_steps=accum_steps, device=device)

run.finish()
print("\nTraining and Validation Complete.")
"""