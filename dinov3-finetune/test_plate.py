import sys
sys.path.insert(0, '../dinov3')
import torch
import dinov3

# Load encoder
weights_path = '/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging/dino_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
encoder = torch.hub.load('../dinov3', 'dinov3_vitl16', source='local', weights=weights_path)
print("Encoder loaded")
print(f"Encoder num_features: {encoder.num_features}")

# Test forward_features
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = encoder.forward_features(x)
    print(f"Output type: {type(out)}")
    if isinstance(out, dict):
        for k, v in out.items():
            if v is not None:
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: None")
    else:
        print(f"Output shape: {out.shape}")

# Now test classification model
from dino_finetune.model.plate_classifier import DINOEncoderLoRAForClassification
model = DINOEncoderLoRAForClassification(
    encoder=encoder,
    r=8,
    emb_dim=encoder.num_features,
    n_classes=85,
    use_lora=True,
    img_dim=(224, 224),
    dropout=0.2,
)
print("Classification model created")
# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,}")
print(f"Trainable params: {trainable:,}")

# Forward pass
logits = model(x)
print(f"Logits shape: {logits.shape}")
print("Test passed.")