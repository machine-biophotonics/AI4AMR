#!/usr/bin/env python3
"""
Evaluate checkpoint at different epochs to analyze grokking behavior.
Also fixes CLS token to be learnable.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_embeddings(embeddings_dir):
    train_emb = np.load(os.path.join(embeddings_dir, 'dinov3_embeddings_train_c512.npz'))
    val_emb = np.load(os.path.join(embeddings_dir, 'dinov3_embeddings_val_c512.npz'))
    test_emb = np.load(os.path.join(embeddings_dir, 'dinov3_embeddings_test_c512.npz'))
    
    X_train, y_train = train_emb['embeddings'], train_emb['labels']
    X_val, y_val = val_emb['embeddings'], val_emb['labels']
    X_test, y_test = test_emb['embeddings'], test_emb['labels']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


class ConvBlock(nn.Module):
    """1D Convolutional block - EXACT copy from original."""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        return x


class HybridCNNTransformer(nn.Module):
    """Hybrid: 1D CNN layers + Transformer - FIXED with learnable CLS token."""
    
    def __init__(self, input_dim=1024, num_classes=85, d_model=256, nhead=4, 
                 num_layers=4, num_cnn_blocks=2, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.cnn_blocks = nn.ModuleList([
            ConvBlock(d_model) for _ in range(num_cnn_blocks)
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # FIXED: Learnable CLS token instead of zeros!
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        
        for cnn in self.cnn_blocks:
            x = cnn(x) + x
        
        # Use learnable CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        x = x[:, 0, :]
        
        return self.output(x)


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    return 100. * correct / total


def main():
    print("="*60)
    print("Evaluating Checkpoints at Different Epochs")
    print("="*60)
    print(f"Device: {device}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings(BASE_DIR)
    num_classes = len(np.unique(y_train))
    print(f"Classes: {num_classes}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    val_loader = DataLoader(val_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=512)
    
    checkpoint_epochs = [100, 200, 300, 500, 800, 1000]
    results = {}
    
    for epoch in checkpoint_epochs:
        ckpt_path = f'checkpoint_hybrid_cnn_trans_2cnn_ep{epoch}.pth'
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            continue
        
        print(f"\n{'='*40}")
        print(f"Loading checkpoint at epoch {epoch}")
        print(f"{'='*40}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Create fresh model and load weights
        model = HybridCNNTransformer(
            input_dim=1024, 
            num_classes=num_classes, 
            d_model=256, 
            nhead=4,
            num_layers=4, 
            num_cnn_blocks=2, 
            dropout=0.3
        ).to(device)
        
        # Load checkpoint - need to handle CLS token mismatch
        # The old model had no cls_token param, so we initialize new one
        state_dict = checkpoint['model_state_dict']
        # Filter out keys that don't match
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        val_acc = evaluate(model, val_loader)
        test_acc = evaluate(model, test_loader)
        
        print(f"Epoch {epoch}: Val Acc = {val_acc:.2f}%, Test Acc = {test_acc:.2f}%")
        
        results[epoch] = {
            'val_acc': val_acc,
            'test_acc': test_acc
        }
    
    # Save results
    print("\n" + "="*60)
    print("GROKKING ANALYSIS RESULTS")
    print("="*60)
    
    for epoch, res in sorted(results.items()):
        print(f"Epoch {epoch:4d}: Val = {res['val_acc']:.2f}%, Test = {res['test_acc']:.2f}%")
    
    with open('grokking_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to grokking_analysis_results.json")


if __name__ == '__main__':
    main()
