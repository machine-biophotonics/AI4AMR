import sys
sys.path.insert(0, '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_max_model')

from mil_model import AttentionMILModel
import torch

print("Testing model...")

# Test attention (256-dim)
m = AttentionMILModel(num_classes=10, pooling_type='attention')
x = torch.randn(2, 25, 3, 224, 224)
out, attn = m(x, return_attention=True)
print(f'Attention: out {out.shape}, attn {attn.shape}')

# Test mean (1280-dim)
m = AttentionMILModel(num_classes=10, pooling_type='mean')
out, attn = m(x, return_attention=True)
print(f'Mean: out {out.shape}, attn {attn.shape}')

# Test max (1280-dim)
m = AttentionMILModel(num_classes=10, pooling_type='max')
out, attn = m(x, return_attention=True)
print(f'Max: out {out.shape}, attn {attn.shape}')

# Test gmp (1280-dim)
m = AttentionMILModel(num_classes=10, pooling_type='gmp')
out, attn = m(x, return_attention=True)
print(f'GMP: out {out.shape}, attn {attn.shape}')

# Test certainty (1280-dim)
m = AttentionMILModel(num_classes=10, pooling_type='certainty')
out, attn = m(x, return_attention=True)
print(f'Certainty: out {out.shape}, attn {attn.shape}')

print("\nAll tests passed!")