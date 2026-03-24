"""
GPU Memory Test Script - Find optimal batch size for fixed grid training.
Tests different batch sizes to maximize GPU utilization.
"""

import torch
from torch import nn
import torchvision.models as models
import numpy as np
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

BASE_DIR = "/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging"

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9
    return 0, 0

def test_forward_pass(batch_size, n_crops=144, img_size=224, channels=3):
    """Test forward pass with given batch size."""
    torch.cuda.reset_peak_memory_stats()
    
    device = torch.device("cuda")
    
    # Create model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1280, 85)  # 85 classes
    )
    model = model.to(device)
    model.eval()
    
    # Create dummy batch
    x = torch.randn(batch_size * n_crops, channels, img_size, img_size).to(device)
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    mem_used, mem_peak = get_gpu_memory()
    
    # Clean up
    del model, x, outputs
    torch.cuda.empty_cache()
    
    return elapsed, mem_used, mem_peak

def main():
    print("=" * 60)
    print("GPU MEMORY TEST FOR FIXED GRID TRAINING")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"\nTest configuration:")
    print(f"  - Image size: 224 x 224")
    print(f"  - Channels: 3 (RGB)")
    print(f"  - Crops per image: 144 (12 x 12 grid)")
    print(f"  - Classes: 85")
    print()
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []
    
    print(f"{'Batch Size':<12} {'Samples/Batch':<15} {'Time (s)':<12} {'Mem Used (GB)':<15} {'Mem Peak (GB)':<15} {'Samples/sec':<12}")
    print("-" * 80)
    
    for bs in batch_sizes:
        try:
            elapsed, mem_used, mem_peak = test_forward_pass(bs)
            samples_per_sec = (bs * 144) / elapsed if elapsed > 0 else 0
            results.append({
                'batch_size': bs,
                'samples': bs * 144,
                'time': elapsed,
                'mem_used': mem_used,
                'mem_peak': mem_peak,
                'samples_per_sec': samples_per_sec
            })
            print(f"{bs:<12} {bs*144:<15} {elapsed:<12.3f} {mem_used:<15.2f} {mem_peak:<15.2f} {samples_per_sec:<12.1f}")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"{bs:<12} {bs*144:<15} OOM ERROR")
            else:
                print(f"{bs:<12} ERROR: {e}")
    
    # Find optimal batch size
    if results:
        print()
        print("=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        
        # Find batch size with best throughput
        best = max(results, key=lambda x: x['samples_per_sec'])
        print(f"Best throughput: batch_size={best['batch_size']} ({best['samples']} samples/batch)")
        print(f"  - Time per batch: {best['time']:.3f}s")
        print(f"  - Samples/sec: {best['samples_per_sec']:.1f}")
        print(f"  - Peak memory: {best['mem_peak']:.2f} GB")
        
        # Find largest batch that fits in memory
        max_batch = max([r for r in results if r['mem_peak'] < 8.0], key=lambda x: x['batch_size'])
        print(f"\nLargest batch that fits (< 8GB): batch_size={max_batch['batch_size']}")
        
        print(f"\nFor training with gradient accumulation (144 crops per image):")
        print(f"  - Use batch_size=1 (1 image per batch = 144 crops)")
        print(f"  - Effective batch = batch_size * n_crops = 144")
        print(f"  - For larger effective batch, use gradient accumulation")

if __name__ == "__main__":
    main()
