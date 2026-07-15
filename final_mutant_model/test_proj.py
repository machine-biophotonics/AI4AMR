#!/usr/bin/env python3
"""Minimal test: does CAAProjection learn at all?"""
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Super-simple: 9 orthogonal class centroids in 1280-D + tiny noise
hd = 1280
n_classes = 9
n_per = 50
batch_size = 50
lr = 3e-3
epochs = 100

rng = np.random.RandomState(42)
# 9 random orthogonal vectors as centroids
M = rng.randn(hd, n_classes)
Q, _ = np.linalg.qr(M)
centroids = Q.T * 3.0  # (9, 1280), norm=3 each, orthogonal

feats, labels = [], []
for c in range(n_classes):
    z = centroids[c] + rng.randn(n_per, hd) * 0.05 * (1 + 0.1*c)
    feats.append(z)
    labels.extend([c] * n_per)
feats = np.concatenate(feats)
feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
labels = np.array(labels)

X = torch.tensor(feats, dtype=torch.float32, device=device)
y = torch.tensor(labels, dtype=torch.long, device=device)

# Model WITHOUT BN
proj = nn.Linear(hd, 128).to(device)
classifier = nn.Linear(128, n_classes).to(device)

optimizer = torch.optim.AdamW(list(proj.parameters()) + list(classifier.parameters()), lr=lr)
ce_loss = nn.CrossEntropyLoss()

# Batch indices
idx = np.arange(n_per * n_classes)

for ep in range(epochs):
    rng.shuffle(idx)
    losses = []
    for i in range(0, len(idx), batch_size):
        bi = idx[i:i+batch_size]
        bx, by = X[bi], y[bi]
        optimizer.zero_grad()
        z = F.normalize(proj(bx), dim=1)
        logits = classifier(z)
        loss = ce_loss(logits, by)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if ep % 20 == 0 or ep == epochs-1:
        print(f"E{ep:3d} CE={np.mean(losses):.4f}")

# Check accuracy
proj.eval()
with torch.no_grad():
    z = F.normalize(proj(X), dim=1)
    logits = classifier(z)
    pred = logits.argmax(1)
    acc = (pred == y).float().mean().item()
print(f"Final acc: {acc*100:.1f}%")
print("If acc > 90%, model CAN learn → toy data generation is the issue")
print("If acc < 30%, there's a bug in the projection/optimization")
