"""
plot_training.py - Generate training plots from saved statistics
Usage: python plot_training.py
"""

import matplotlib
matplotlib.use('Agg')
import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_stats():
    """Load training statistics from JSON"""
    stats_file = os.path.join(BASE_DIR, 'training_stats.json')
    
    if not os.path.exists(stats_file):
        print(f"ERROR: {stats_file} not found!")
        print("Run training first to generate statistics.")
        return None
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    return stats

def plot_training_curves(stats):
    """Generate training curves plot"""
    import matplotlib.pyplot as plt
    
    epochs = stats['epochs']
    train_loss = stats['train_loss']
    train_acc = stats['train_acc']
    val_loss = stats['val_loss']
    val_acc = stats['val_acc']
    lr = stats['lr']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, lr, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    gap = [t - v for t, v in zip(train_acc, val_acc)]
    axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap (%)')
    axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('BacNet Training Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, 'training_plots_bacnet.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def plot_interactive(stats, output_html='training_plots_interactive.html'):
    """Generate interactive HTML plots using Plotly"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return None
    
    epochs = stats['epochs']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss Curves', 'Accuracy Curves', 'Learning Rate', 'Train-Val Gap'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=stats['train_loss'], name='Train Loss', 
                   line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=stats['val_loss'], name='Val Loss',
                   line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=stats['train_acc'], name='Train Acc',
                   line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=stats['val_acc'], name='Val Acc',
                   line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=stats['lr'], name='LR',
                   line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    gap = [t - v for t, v in zip(stats['train_acc'], stats['val_acc'])]
    fig.add_trace(
        go.Scatter(x=epochs, y=gap, name='Gap',
                   line=dict(color='purple', width=2),
                   fill='tozeroy'),
        row=2, col=2
    )
    
    fig.update_layout(
        title='BacNet Training Statistics (Interactive)',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text='Epoch', row=2, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=2)
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy (%)', row=1, col=2)
    fig.update_yaxes(title_text='LR', row=2, col=1)
    fig.update_yaxes(title_text='Gap (%)', row=2, col=2)
    
    output_path = os.path.join(BASE_DIR, output_html)
    fig.write_html(output_path)
    print(f"Saved interactive plot: {output_path}")
    
    return output_path

def print_summary(stats):
    """Print training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(stats['epochs'])}")
    print(f"Best val accuracy: {max(stats['val_acc']):.2f}% (epoch {stats['val_acc'].index(max(stats['val_acc'])) + 1})")
    print(f"Final train accuracy: {stats['train_acc'][-1]:.2f}%")
    print(f"Final val accuracy: {stats['val_acc'][-1]:.2f}%")
    print(f"Final train loss: {stats['train_loss'][-1]:.4f}")
    print(f"Final val loss: {stats['val_loss'][-1]:.4f}")
    print(f"Final LR: {stats['lr'][-1]:.6f}")
    
    gap = stats['train_acc'][-1] - stats['val_acc'][-1]
    print(f"\nFinal overfitting gap: {gap:.2f}%")
    print("="*60 + "\n")

def main():
    print("="*60)
    print("BacNet Training Plot Generator")
    print("="*60)
    
    stats = load_stats()
    if stats is None:
        return
    
    print_summary(stats)
    
    plot_training_curves(stats)
    interactive_path = plot_interactive(stats)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Static plot: training_plots_bacnet.png")
    if interactive_path:
        print(f"Interactive plot: {interactive_path}")
    print("="*60)

if __name__ == "__main__":
    main()
