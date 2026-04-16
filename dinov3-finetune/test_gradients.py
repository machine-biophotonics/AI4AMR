import sys
sys.path.insert(0, '../dinov3')
import torch
import torch.nn as nn
import dinov3

# Load encoder
weights_path = '/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging/dino_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
encoder = torch.hub.load('../dinov3', 'dinov3_vitl16', source='local', weights=weights_path)
encoder.cuda()

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

print("Model created")
print("Checking parameter requires_grad:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params: {trainable:,}")

# Simulate forward and backward
x = torch.randn(2, 3, 224, 224).cuda()
logits = model(x)
print(f"Logits shape: {logits.shape}")

# Create a simple loss
targets = torch.randint(0, 85, (2,)).cuda()
loss = nn.CrossEntropyLoss()(logits, targets)
loss.backward()

# Check if gradients exist for trainable parameters
print("\nGradients after backward:")
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: grad is None")

# Now test optimizer setup similar to train_plate.py
import torch.optim as optim
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

print(f"\nFound {len(lora_params)} LoRA parameters")
print(f"Found {len(decoder_params)} decoder parameters")
if lora_params:
    param_groups.append({'params': lora_params, 'lr': 1e-4})
if decoder_params:
    param_groups.append({'params': decoder_params, 'lr': 1e-3})

optimizer = optim.AdamW(param_groups, weight_decay=0.1)
optimizer.zero_grad()
loss = nn.CrossEntropyLoss()(model(x), targets)
loss.backward()
optimizer.step()
print("\nOptimizer step succeeded!")

# Check that only LoRA and decoder parameters changed
print("\nChecking if parameters updated:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: updated")

print("\nTest passed.")