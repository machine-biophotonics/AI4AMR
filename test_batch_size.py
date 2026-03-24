"""
Test script to find optimal batch size and worker settings for training.
"""
import torch
import time
import os
import sys

BASE_DIR = "/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging"
os.chdir(BASE_DIR)

# Import necessary modules
from train import (
    train_data, val_data, test_data,
    model, criterion, optimizer, scheduler,
    SEED, device
)

def test_batch_size(batch_size: int, num_workers: int = 4) -> dict:
    """Test a specific batch size configuration."""
    from torch.utils.data import DataLoader
    
    torch.cuda.empty_cache()
    
    # Create dataloader with given batch size
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Move model to GPU
    model = torch.nn.Sequential(
        torch.nn.Linear(1280, 97)
    ).to(device)
    
    # Test forward pass
    total_patches = 0
    start_time = time.time()
    max_batches = 10  # Test only first 10 batches
    
    try:
        for i, (patches, labels, _) in enumerate(loader):
            if i >= max_batches:
                break
                
            batch_size_actual, n_patches, C, H, W = patches.shape
            total_patches += batch_size_actual * n_patches
            
            # Simulate forward pass
            patches = patches.view(-1, C, H, W).to(device, non_blocking=True)
            outputs = torch.randn(batch_size_actual, n_patches, 97).to(device)
            loss = outputs.mean()
            loss.backward()
            
            torch.cuda.synchronize()
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                'batch_size': batch_size,
                'num_workers': num_workers,
                'status': 'OOM',
                'patches_per_batch': batch_size * train_data.n_patches,
                'error': str(e)
            }
        raise
    
    elapsed = time.time() - start_time
    throughput = total_patches / elapsed if elapsed > 0 else 0
    
    del model, loader, patches, outputs, loss
    torch.cuda.empty_cache()
    
    return {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'status': 'OK',
        'patches_per_batch': batch_size * train_data.n_patches,
        'time_per_batch': elapsed / max_batches if max_batches > 0 else 0,
        'throughput': throughput,
    }

def find_optimal_batch_size():
    """Test different batch sizes to find optimal."""
    print("=" * 60)
    print("BATCH SIZE OPTIMIZATION TEST")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Image patches per sample: {train_data.n_patches}")
    print()
    
    batch_sizes = [2, 4, 6, 8, 10, 12, 16]
    results = []
    
    for bs in batch_sizes:
        print(f"\nTesting batch_size={bs}...", end=" ", flush=True)
        result = test_batch_size(bs)
        results.append(result)
        
        if result['status'] == 'OOM':
            print(f"OOM! (patches/batch: {result['patches_per_batch']})")
        else:
            print(f"OK (patches/batch: {result['patches_per_batch']}, "
                  f"throughput: {result['throughput']:.0f} patches/sec)")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    valid_results = [r for r in results if r['status'] == 'OK']
    if valid_results:
        # Find best by throughput
        best = max(valid_results, key=lambda x: x['throughput'])
        print(f"\nBest batch size by throughput: {best['batch_size']}")
        print(f"  - Patches per batch: {best['patches_per_batch']}")
        print(f"  - Throughput: {best['throughput']:.0f} patches/sec")
        
        # Find largest batch size that's still valid
        largest_valid = max(valid_results, key=lambda x: x['batch_size'])
        print(f"\nLargest valid batch size: {largest_valid['batch_size']}")
        print(f"  - Recommended for training (balances speed vs. gradient quality)")
        
        return largest_valid['batch_size']
    else:
        print("No valid batch sizes found!")
        return 4

if __name__ == "__main__":
    optimal_bs = find_optimal_batch_size()
    print(f"\nRecommended batch size: {optimal_bs}")
