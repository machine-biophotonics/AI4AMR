import sys
sys.path.insert(0, '../dinov3')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dinov3

# Load encoder
weights_path = '/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging/dino_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
encoder = torch.hub.load('../dinov3', 'dinov3_vitl16', source='local', weights=weights_path).cuda()

from dino_finetune.model.plate_classifier import DINOEncoderLoRAForClassification
model = DINOEncoderLoRAForClassification(
    encoder=encoder,
    r=8,
    emb_dim=encoder.num_features,
    n_classes=85,
    use_lora=True,
    img_dim=(224, 224),
    dropout=0.2,
).cuda()

# Dataset
from dino_finetune.plate_dataset import create_datasets
train_dataset, val_dataset, test_dataset = create_datasets(
    data_root='/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging',
    label_json_path='/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/plate maps/plate_well_id_path.json',
    stain_augmentation=False,
    target_size=(224, 224),
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

# Optimizer with correct param grouping
param_groups = []
lora_params = []
decoder_params = []
for name, param in model.named_parameters():
    if 'linear_a_q' in name or 'linear_b_q' in name or 'linear_a_v' in name or 'linear_b_v' in name:
        lora_params.append(param)
    elif 'decoder' in name:
        decoder_params.append(param)
    else:
        param.requires_grad = False
if lora_params:
    param_groups.append({'params': lora_params, 'lr': 1e-4})
if decoder_params:
    param_groups.append({'params': decoder_params, 'lr': 1e-3})
optimizer = optim.AdamW(param_groups, weight_decay=0.1)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# Train one batch
model.train()
for images, labels, plates in train_loader:
    images = images.float().cuda()
    labels = labels.long().cuda()
    optimizer.zero_grad()
    logits = model(images)
    loss = focal_loss(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"Batch loss: {loss.item():.4f}")
    break

print("Single batch training succeeded!")

# Validate one batch
model.eval()
with torch.no_grad():
    for images, labels, plates in train_loader:
        images = images.float().cuda()
        labels = labels.long().cuda()
        logits = model(images)
        loss = focal_loss(logits, labels)
        _, predicted = torch.max(logits, 1)
        acc = (predicted == labels).float().mean()
        print(f"Validation batch loss: {loss.item():.4f}, acc: {acc.item():.4f}")
        break

print("All tests passed.")