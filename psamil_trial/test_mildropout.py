import sys
sys.path.insert(0, "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/psamil_trial")
from mil_model import Mildropout
import torch

d = Mildropout(topk=3, kernel=7)
d.train()
x = torch.randn(4, 25, 1280)
y = d(x)
print(f"psamil_trial Mildropout: {x.shape} -> {y.shape}")

d.eval()
y2 = d(x)
print(f"eval mode: {y2.shape}")
END_OFFER
