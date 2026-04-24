import torch
import sys
sys.path.insert(0, '.')
from src.mil_model import AttentionMILModel
from src.mildropout import Mildropout

print('=== FINAL PSAMIL_TRIAL TEST ===')

# Test 1: Model without Mildropout
print('Test 1: Model without Mildropout')
m1 = AttentionMILModel(num_classes=96)
x = torch.randn(2, 25, 3, 224, 224)
o1, a1 = m1(x, return_attention=True)
print(f'  out={o1.shape}, attn={a1.shape} OK')

# Test 2: Model with Mildropout
print('Test 2: Model with Mildropout')
m2 = AttentionMILModel(num_classes=96, mildropout=Mildropout(topk=3, kernel=7))
m2.train()
o2, a2 = m2(x, return_attention=True)
print(f'  out={o2.shape}, attn={a2.shape} OK')

# Test 3: Mildropout alone
print('Test 3: Mildropout alone')
d = Mildropout(topk=3, kernel=7)
d.train()
feat = torch.randn(4, 25, 1280)
feat_out = d(feat)
print(f'  {feat.shape} -> {feat_out.shape} OK')

# Test 4: Eval mode (no dropout)
print('Test 4: Mildropout eval mode')
d.eval()
feat_out2 = d(feat)
print(f'  {feat.shape} -> {feat_out2.shape} OK')

# Test 5: topk=0 (disabled)
print('Test 5: Mildropout topk=0 (disabled)')
d0 = Mildropout(topk=0)
d0.train()
feat_out0 = d0(feat)
print(f'  {feat.shape} -> {feat_out0.shape} OK')

print('')
print('=== ALL TESTS PASSED ===')
