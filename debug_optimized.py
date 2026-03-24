"""
Debug script for train_ai4ab_optimized.py
Tests 1 batch to verify all optimizations work correctly
"""

import os
import sys
import torch
from torch import nn
import torchvision.models as models
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test 1: Check AMP - using torch.autocast with device_type (recommended by docs)
print("\n" + "="*50)
print("TEST 1: Mixed Precision (AMP)")
print("="*50)
try:
    scaler = torch.amp.GradScaler('cuda')
    print("✓ GradScaler initialized successfully")
except Exception as e:
    print(f"✗ GradScaler failed: {e}")

# Test 2: Check model creation and channels_last
print("\n" + "="*50)
print("TEST 2: Model + Channels_last")
print("="*50)
model = None  # Initialize to avoid "possibly unbound" errors
try:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    print("✓ Model created and converted to channels_last")
except Exception as e:
    print(f"✗ Model failed: {e}")

# Test 3: Check forward pass with AMP
print("\n" + "="*50)
print("TEST 3: Forward Pass with AMP")
print("="*50)
try:
    model.eval()
    dummy_input = torch.randn(16, 3, 256, 256, device=device)
    dummy_input = dummy_input.to(memory_format=torch.channels_last)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")

# Test 4: Check backward pass with GradScaler
print("\n" + "="*50)
print("TEST 4: Backward Pass with GradScaler")
print("="*50)
try:
    model.train()
    dummy_input = torch.randn(16, 3, 256, 256, device=device)
    dummy_input = dummy_input.to(memory_format=torch.channels_last)
    dummy_target = torch.randint(0, 1000, (16,), device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"✓ Backward pass successful. Loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Backward pass failed: {e}")

# Test 5: Check torch.compile
print("\n" + "="*50)
print("TEST 5: torch.compile()")
print("="*50)
try:
    model_simple = models.efficientnet_b0(weights=None)
    model_simple = model_simple.to(device)
    
    compile_start = time.time()
    model_compiled = torch.compile(model_simple, mode="reduce-overhead")
    compile_time = time.time() - compile_start
    
    # Test inference with compiled model
    with torch.no_grad():
        _ = model_compiled(torch.randn(4, 3, 256, 256, device=device))
    
    print(f"✓ torch.compile successful in {compile_time:.1f}s")
except Exception as e:
    print(f"✗ torch.compile failed: {e}")
    print("  (This is OK if PyTorch version < 2.0)")

# Test 6: Speed benchmark
print("\n" + "="*50)
print("TEST 6: Speed Benchmark")
print("="*50)
try:
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(torch.randn(32, 3, 256, 256, device=device))
    
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 50
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(torch.randn(32, 3, 256, 256, device=device))
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    images_per_sec = (iterations * 32) / elapsed
    print(f"✓ Throughput: {images_per_sec:.1f} images/sec ({elapsed/iterations*1000:.1f} ms/batch)")
    
except Exception as e:
    print(f"✗ Benchmark failed: {e}")

# Test 7: Multi-crop input (like AI4AB)
print("\n" + "="*50)
print("TEST 7: Multi-Crop Input (9 crops)")
print("="*50)

class AvgPoolCNN(nn.Module):
    def __init__(self, num_classes=1000, n_crops=9):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier[1] = nn.Identity()
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.avg_pool_1d = nn.AvgPool1d(kernel_size=n_crops)
        self.fc_final = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        bs, ncrops, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.backbone.features(x)
        features = self.avg_pool_2d(features).view(bs, ncrops, -1)
        feat_vec = self.avg_pool_1d(features.permute(0, 2, 1)).view(bs, -1)
        logits = self.fc_final(feat_vec)
        return logits, feat_vec

try:
    multi_crop_model = AvgPoolCNN(num_classes=1000, n_crops=9)
    multi_crop_model = multi_crop_model.to(device)
    
    # 16 images, 9 crops each, 3 channels, 256x256
    multi_crop = torch.randn(16, 9, 3, 256, 256, device=device)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, feat_vec = multi_crop_model(multi_crop)
    
    print(f"✓ Multi-crop forward pass successful")
    print(f"  Input: {multi_crop.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Features: {feat_vec.shape}")
    
except Exception as e:
    print(f"✗ Multi-crop test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Full training step with compiled model
print("\n" + "="*50)
print("TEST 8: Full Training Step with Compiled Model")
print("="*50)
try:
    compiled_model = AvgPoolCNN(num_classes=100, n_crops=9)
    compiled_model = compiled_model.to(device)
    compiled_model = torch.compile(compiled_model, mode="reduce-overhead")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    # One training step
    multi_crop = torch.randn(8, 9, 3, 256, 256, device=device)
    labels = torch.randint(0, 100, (8,), device=device)
    
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logits, _ = compiled_model(multi_crop)
        loss = criterion(logits, labels)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    print(f"✓ Full training step successful. Loss: {loss.item():.4f}")
    print(f"  Note: Compiled model type: {type(compiled_model)}")
    
except Exception as e:
    print(f"✗ Full training step failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Check saving/loading compiled model
print("\n" + "="*50)
print("TEST 9: Save/Load with Compiled Model")
print("="*50)
try:
    # When saving a compiled model, need to save the underlying module
    test_model = AvgPoolCNN(num_classes=50, n_crops=9)
    test_model = test_model.to(device)
    test_model = torch.compile(test_model, mode="reduce-overhead")
    
    # Save the state dict (works for both compiled and uncompiled)
    checkpoint = {
        'model_state_dict': test_model.state_dict(),
        'epoch': 1
    }
    torch.save(checkpoint, '/tmp/test_compiled.pth')
    
    # Load into a new uncompiled model
    loaded_model = AvgPoolCNN(num_classes=50, n_crops=9)
    loaded_model = loaded_model.to(device)
    loaded_model.load_state_dict(torch.load('/tmp/test_compiled.pth', weights_only=True)['model_state_dict'])
    
    # Test inference
    with torch.no_grad():
        out = loaded_model(torch.randn(2, 9, 3, 256, 256, device=device))[0]
    
    print(f"✓ Save/Load successful. Output shape: {out.shape}")
    
except Exception as e:
    print(f"✗ Save/Load failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("ALL TESTS COMPLETED")
print("="*50)
