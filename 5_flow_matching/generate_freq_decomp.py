#!/usr/bin/env python3
"""Load a trained FreqFlow model, generate a sample from a random class,
and decompose it into low-frequency + high-frequency components.

Usage:
    python3 generate_freq_decomp.py
    python3 generate_freq_decomp.py --checkpoint path/to/flow_best.pth --class_id 42
"""
import os, sys, argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from flow_model import FreqFlowUNet, FlowUNet, Fourier_filter, sample


def load_model_from_checkpoint(ckpt_path: str, device: str = 'cuda'):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']

    if args.get('freq_flow', False):
        block_channels = tuple(int(x) for x in args['block_channels'].split(','))
        freq_block_channels = tuple(int(x) for x in args.get('freq_block_channels', args['block_channels']).split(','))
        num_class_embeds = len(args.get('class_names', [])) or 185
        model = FreqFlowUNet(
            in_channels=1,
            sample_size=224,
            block_out_channels=block_channels,
            freq_block_out_channels=freq_block_channels,
            layers_per_block=2,
            num_class_embeds=num_class_embeds,
            freq_filter_D=args.get('freq_filter_D', 8.0),
        ).to(device)
        print(f"  Loaded FreqFlowUNet: spatial={block_channels}, freq={freq_block_channels}")
    else:
        block_channels = tuple(int(x) for x in args['block_channels'].split(','))
        num_class_embeds = len(args.get('class_names', [])) or 185
        model = FlowUNet(
            in_channels=1, sample_size=224,
            block_out_channels=block_channels, layers_per_block=2,
            num_class_embeds=num_class_embeds,
        ).to(device)
        print(f"  Loaded FlowUNet: channels={block_channels}")

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, args


def main():
    parser = argparse.ArgumentParser(description='Freq decomposition of generated samples')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to flow_best.pth (auto-detect if omitted)')
    parser.add_argument('--class_id', type=int, default=None,
                        help='Class index to generate (random if omitted)')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='ODE integration steps')
    parser.add_argument('--freq_filter_D', type=float, default=None,
                        help='Override Gaussian filter cutoff D')
    parser.add_argument('--output', type=str, default='freq_decomposition.png',
                        help='Output image path')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate and show')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Auto-discover best checkpoint
    if args.checkpoint is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_dirs = sorted([d for d in os.listdir(script_dir)
                          if d.startswith('flow_run_') and os.path.isdir(os.path.join(script_dir, d))])
        best_path = None
        for rd in reversed(run_dirs):
            candidate = os.path.join(script_dir, rd, 'flow_best.pth')
            if os.path.exists(candidate):
                best_path = candidate
                break
        if best_path is None:
            print("No flow_best.pth found. Specify --checkpoint manually.")
            sys.exit(1)
        args.checkpoint = best_path
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    print("Loading model ...")
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)
    freq_flow = model_args.get('freq_flow', False)
    delta_fm = model_args.get('delta_fm', False)
    D = args.freq_filter_D if args.freq_filter_D is not None else model_args.get('freq_filter_D', 8.0)
    num_classes = model_args.get('num_classes', 185)

    # Load class names
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, script_dir)
    from mil_model import load_labels
    _, class_names, _ = load_labels(project_root, script_dir)
    num_classes = len(class_names)

    # Pick random class(es)
    if args.class_id is not None:
        class_ids = [args.class_id]
    else:
        class_ids = random.sample(range(num_classes), min(args.num_samples, num_classes))

    # Generate
    print(f"Generating {len(class_ids)} sample(s) ...")
    samples = []
    for cid in class_ids:
        c_tensor = torch.tensor([cid], device=device)
        x_gen = sample(model, 1, num_steps=args.num_steps,
                        class_labels=c_tensor, device=device,
                        freq_flow=freq_flow)
        samples.append(x_gen.cpu())

    # Decompose and plot
    n = len(samples)
    fig = plt.figure(figsize=(n * 4, 5))
    gs = GridSpec(3, n, figure=fig, hspace=0.15, wspace=0.1)

    for i, (cid, x_gen) in enumerate(zip(class_ids, samples)):
        img = x_gen.squeeze()  # (H, W) in [-1, 1]
        img_01 = (img * 0.5 + 0.5).clamp(0, 1)  # [0, 1]

        # Fourier decomposition
        x_low, x_high = Fourier_filter(img_01.unsqueeze(0).unsqueeze(0), D)
        x_low = x_low.squeeze()
        x_high = x_high.squeeze()

        name = class_names[cid] if cid < len(class_names) else f'class_{cid}'

        # Row 0: combined
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(img_01, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{name}', fontsize=8)
        ax.set_ylabel('Combined', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 1: low-frequency
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(x_low, cmap='gray', vmin=0, vmax=1)
        ax.set_ylabel('Low-freq', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 2: high-frequency (stretched for visibility)
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(x_high, cmap='gray', vmin=0, vmax=1)
        ax.set_ylabel('High-freq', fontsize=7)
        ax.set_xlabel(f'D={D}', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'FreqFlow Decomposition (D={D})', fontsize=9, y=0.98)
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
