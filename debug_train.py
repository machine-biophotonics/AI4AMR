"""Debug script to verify train.py code integrity"""

import sys
sys.path.insert(0, '/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging')

print("=" * 60)
print("DEBUG: Verifying train.py code integrity")
print("=" * 60)

# Test 1: Import all modules
print("\n[1] Testing imports...")
try:
    import train
    print("    ✓ All imports successful")
except Exception as e:
    print(f"    ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check classes exist
print("\n[2] Checking classes...")
classes_to_check = [
    'MixedCropDataset', 'ShuffledCropSampler', 'EarlyStopping',
    'unfreeze_by_epoch', 'rebuild_optimizer_with_differential_lr'
]
for cls in classes_to_check:
    if hasattr(train, cls):
        print(f"    ✓ {cls} exists")
    else:
        print(f"    ✗ {cls} missing")

# Test 3: Check functions
print("\n[3] Checking functions...")
funcs = [
    'unfreeze_by_epoch', 'rebuild_optimizer_with_differential_lr',
    'reset_unfreeze_state', 'evaluate', 'get_all_predictions_and_labels'
]
for fn in funcs:
    if hasattr(train, fn):
        print(f"    ✓ {fn} exists")
    else:
        print(f"    ✗ {fn} missing")

# Test 4: Check model creation
print("\n[4] Testing model creation...")
try:
    import torch
    from train import model
    print(f"    ✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
except Exception as e:
    print(f"    ✗ Model creation error: {e}")

# Test 5: Check optimizer
print("\n[5] Testing optimizer...")
try:
    from train import optimizer
    print(f"    ✓ Optimizer created with {len(optimizer.param_groups)} groups")
except Exception as e:
    print(f"    ✗ Optimizer error: {e}")

# Test 6: Check data loaders
print("\n[6] Testing data loaders...")
try:
    from train import train_loader, val_loader, test_loader
    print(f"    ✓ Train loader: {len(train_loader)} batches")
    print(f"    ✓ Val loader: {len(val_loader)} batches")
    print(f"    ✓ Test loader: {len(test_loader)} batches")
except Exception as e:
    print(f"    ✗ DataLoader error: {e}")

# Test 7: Check AMP
print("\n[7] Testing AMP...")
try:
    import torch
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(4, 3, 224, 224).cuda()
    with torch.amp.autocast('cuda'):
        y = x.mean()
    print(f"    ✓ AMP works: output dtype = {y.dtype}")
except Exception as e:
    print(f"    ✗ AMP error: {e}")

# Test 8: Check grad zeroing
print("\n[8] Testing gradient zeroing...")
try:
    import torch
    t = torch.randn(10, requires_grad=True).cuda()
    opt = torch.optim.SGD([t], lr=0.1)
    loss = t.sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f"    ✓ zero_grad(set_to_none=True) works")
except Exception as e:
    print(f"    ✗ zero_grad error: {e}")

# Test 9: Check reproducibility settings
print("\n[9] Checking reproducibility...")
print(f"    ✓ torch.backends.cudnn.deterministic = {torch.backends.cudnn.deterministic}")
print(f"    ✓ torch.backends.cudnn.benchmark = {torch.backends.cudnn.benchmark}")
print(f"    ✓ SEED = {train.SEED}")

# Test 10: Check command line args
print("\n[10] Checking CLI arguments...")
try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_crops', type=int, default=144)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(['--n_crops', '9', '--epochs', '1'])
    print(f"    ✓ CLI args: n_crops={args.n_crops}, epochs={args.epochs}")
except Exception as e:
    print(f"    ✗ CLI error: {e}")

# Test 11: Quick forward pass
print("\n[11] Testing forward pass...")
try:
    import torch
    from train import model
    
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 224, 224).cuda()
        with torch.amp.autocast('cuda'):
            out = model(x)
        print(f"    ✓ Forward pass: input {x.shape} -> output {out.shape}")
        print(f"    ✓ Output classes: {out.shape[-1]}")
except Exception as e:
    print(f"    ✗ Forward pass error: {e}")

# Test 12: Check unfreeze state
print("\n[12] Testing gradual unfreezing...")
try:
    import torch
    from train import model, unfreeze_by_epoch, reset_unfreeze_state, _unfrozen_state
    
    # Reset and test
    reset_unfreeze_state()
    unfreeze_by_epoch(model, epoch=0)
    
    # Check what's frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    ✓ Epoch 0 (start): {trainable:,} / {total:,} trainable params ({100*trainable/total:.1f}%)")
    
    # Test epoch 1
    unfreeze_by_epoch(model, epoch=1)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    ✓ Epoch 1: {trainable:,} trainable (classifier only)")
    
except Exception as e:
    print(f"    ✗ Unfreeze error: {e}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
