#!/usr/bin/env python3
"""Toy CAA: 3 groups in 5-D latent → project to 1280-D + optimize CAA loss.

Groups are at 0°, 120°, 240° in 5-D. Small noise in latent space (not 1280-D)
so within-class variance stays low. KO→drug = small rotation in latent space.
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from supcon_loss import SupConLoss

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def uniformity_loss(z, t=2.0):
    N = z.size(0)
    if N < 2: return torch.tensor(0.0, device=z.device)
    sim = torch.mm(z, z.t())
    mask = ~torch.eye(N, dtype=bool, device=z.device)
    pairwise_l2_sq = 2 - 2 * sim[mask].view(N, N - 1)
    return torch.log(pairwise_l2_sq.mul(-t).exp().mean())

def gromov_wasserstein_loss(v_m, v_d, reg=0.05, outer_iter=10, inner_iter=20):
    N, M = len(v_m), len(v_d)
    if N < 2 or M < 2: return torch.tensor(0.0, device=v_m.device)
    C_m = torch.cdist(v_m, v_m, p=2).pow(2)
    C_d = torch.cdist(v_d, v_d, p=2).pow(2)
    C_m_sq, C_d_sq = C_m ** 2, C_d ** 2
    p = torch.ones(N, device=v_m.device) / N
    q = torch.ones(M, device=v_m.device) / M
    with torch.no_grad():
        T = p.unsqueeze(1) * q.unsqueeze(0)
        for _ in range(outer_iter):
            rs = T.sum(dim=1, keepdim=True)
            cs = T.sum(dim=0, keepdim=True)
            cross = C_m @ T @ C_d.T
            cost = C_m_sq @ rs + cs @ C_d_sq.T - 2 * cross
            K = torch.exp(-cost / max(reg, 1e-8))
            u = torch.ones(N, device=v_m.device)
            for _ in range(inner_iter):
                v = q / (K.T @ u + 1e-8)
                u = p / (K @ v + 1e-8)
            T = u.unsqueeze(1) * K * v.unsqueeze(0)
        T_star = T.detach()
    return (C_m_sq @ p) @ p + (C_d_sq @ q) @ q - 2 * torch.trace(C_m @ T_star @ C_d @ T_star.T)


class CAAProjection(nn.Module):
    def __init__(self, in_features=1280, embed_dim=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, embed_dim), nn.BatchNorm1d(embed_dim))
    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--lambda_gw', type=float, default=1.0)
parser.add_argument('--lambda_struct', type=float, default=1.0)
parser.add_argument('--lambda_unif', type=float, default=1.0)
parser.add_argument('--lambda_ce', type=float, default=1.0)
parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--n_per_class', type=int, default=100)
parser.add_argument('--n_anchors', type=int, default=300)
parser.add_argument('--plot_every', type=int, default=50)
parser.add_argument('--output_dir', type=str, default='toy_caa')
args = parser.parse_args()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HD = 1280     # high-dim (EfficientNet output)
LD = 5        # latent dim for data generation
NG = 3        # groups
NPG = 3       # classes per group


def create_latent_projection(d_in, d_out, seed=42):
    """Fixed random orthogonal projection from d_in → d_out."""
    rng = np.random.RandomState(seed)
    W = rng.randn(d_out, d_in)
    Q, _ = np.linalg.qr(W)
    return torch.tensor(Q.T, dtype=torch.float32)  # (d_in, d_out)


def create_toy_data():
    """Generate data in 5-D latent space, project to 1280-D."""
    n = args.n_per_class
    na = args.n_anchors
    rng = np.random.RandomState(42)

    # --- Latent group prototypes at 120° separation in first 2 dims ---
    angles = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
    proto = np.zeros((NG, LD))
    proto[:, 0] = np.cos(angles) * 3.0
    proto[:, 1] = np.sin(angles) * 3.0

    # --- KO→drug rotation in LD latent space (small angle) ---
    theta = 0.15  # rad ~8.6°
    c, s = np.cos(theta), np.sin(theta)
    rot = np.eye(LD)
    rot[:2, :2] = [[c, -s], [s, c]]
    drug_proto = proto @ rot.T

    # --- Latent features with small noise, then project to HD ---
    proj = create_latent_projection(LD, HD)  # (LD, HD)

    def make_latent_feats(protos, noise_std=0.15):
        feats = []
        for g in range(NG):
            for c in range(NPG):
                z = protos[g] + rng.randn(n, LD) * noise_std * (1 + 0.1*c)
                feats.append(z)
        return np.concatenate(feats, axis=0)

    def latent_to_hd(latent):
        feats = latent @ proj.numpy()  # (N, HD)
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        return feats

    z_ko = make_latent_feats(proto)
    z_drug = make_latent_feats(drug_proto)

    ko_feats = latent_to_hd(z_ko)
    drug_feats = latent_to_hd(z_drug)

    # Labels
    ko_labels = np.concatenate([np.full(n, g*NPG + c) for g in range(NG) for c in range(NPG)])
    drug_labels = ko_labels.copy()

    # --- Random anchors (on hypersphere in HD) ---
    def rand_sphere(n_):
        x = torch.randn(n_, HD)
        return (x / x.norm(dim=1, keepdim=True)).numpy()

    ancA = rand_sphere(na)
    ancB = rand_sphere(na)
    bridgeA = rand_sphere(na)
    bridgeB = rand_sphere(na)

    # --- Package ---
    m_all = np.concatenate([ancA, ko_feats], axis=0)
    m_lbl = np.concatenate([np.full(na, 0), ko_labels + 1])
    m_ctrl_group = {'A': [0]}
    m_classes = ['anchor_A'] + [f'g{g}c{c}' for g in range(NG) for c in range(NPG)]

    c_all = np.concatenate([bridgeA, bridgeB], axis=0)
    c_lbl = np.concatenate([np.full(na, 0), np.full(na, 1)])
    c_ctrl_group = {'A': [0], 'B': [1]}
    c_classes = ['anchor_A', 'anchor_B']

    n_nc = NG * NPG
    d_all = np.concatenate([drug_feats, ancB], axis=0)
    d_lbl = np.concatenate([drug_labels + 1, np.full(na, n_nc + 1)])
    d_ctrl_group = {'B': [n_nc + 1]}
    d_classes = [f'g{g}c{c}' for g in range(NG) for c in range(NPG)] + ['anchor_B']

    print(f"Toy data ({LD}-D latent → {HD}-D via fixed projection):")
    print(f"  Mutant: {len(m_all)}  ({na} anchors + {n_nc*n} KO, {n_nc} classes)")
    print(f"  Drug:   {len(d_all)}  ({n_nc*n} drugs + {na} anchors, {n_nc} classes)")
    print(f"  Bridge: {len(c_all)}  (A: {na}, B: {na})")
    print(f"  3 groups at 120° in {LD}-D, drug rotation {theta*180/np.pi:.1f}°")

    return {
        'm_feats': torch.tensor(m_all, dtype=torch.float32),
        'm_lbl': torch.tensor(m_lbl, dtype=torch.long),
        'm_classes': m_classes, 'm_ctrl_group': m_ctrl_group,
        'c_feats': torch.tensor(c_all, dtype=torch.float32),
        'c_lbl': torch.tensor(c_lbl, dtype=torch.long),
        'c_classes': c_classes, 'c_ctrl_group': c_ctrl_group,
        'd_feats': torch.tensor(d_all, dtype=torch.float32),
        'd_lbl': torch.tensor(d_lbl, dtype=torch.long),
        'd_classes': d_classes, 'd_ctrl_group': d_ctrl_group,
    }


def plot_tsne(z, m_lbl, d_lbl, m_ctrl_t, d_ctrl_t, epoch, path):
    m_nc = ~torch.isin(m_lbl, m_ctrl_t) if len(m_ctrl_t) else torch.ones_like(m_lbl, dtype=torch.bool)
    d_nc = ~torch.isin(d_lbl, d_ctrl_t) if len(d_ctrl_t) else torch.ones_like(d_lbl, dtype=torch.bool)

    z_m_nc = z[m_nc].cpu().numpy()
    z_d_nc = z[d_nc].cpu().numpy()
    z_m_ctrl = z[~m_nc].cpu().numpy() if (~m_nc).any() else np.empty((0, 2))
    z_d_ctrl = z[~d_nc].cpu().numpy() if (~d_nc).any() else np.empty((0, 2))

    all_z = np.concatenate([z_m_nc, z_d_nc, z_m_ctrl, z_d_ctrl], axis=0)
    n_m, n_d = len(z_m_nc), len(z_d_nc)

    if len(all_z) < 10: return

    ppl = min(40, max(5, len(all_z) // 3))
    tsne = TSNE(n_components=2, perplexity=ppl, random_state=42, init='random', learning_rate='auto')
    xy = tsne.fit_transform(all_z)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    lbl_m_nc = m_lbl[m_nc]
    for g in range(NG):
        for c in range(NPG):
            mask = torch.isin(lbl_m_nc, torch.tensor([g*NPG + c + 1], device=m_lbl.device)).cpu().numpy()
            pts = xy[:n_m][mask]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=colors[g], marker='o', alpha=0.5, s=15)

    lbl_d_nc = d_lbl[d_nc]
    for g in range(NG):
        for c in range(NPG):
            mask = torch.isin(lbl_d_nc, torch.tensor([g*NPG + c + 1], device=d_lbl.device)).cpu().numpy()
            pts = xy[n_m:n_m + n_d][mask]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=colors[g], marker='x', alpha=0.5, s=15)

    if len(xy) > n_m + n_d:
        ax.scatter(xy[n_m+n_d:, 0], xy[n_m+n_d:, 1], c='gray', marker='s', alpha=0.3, s=10, label='Anchors')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=6, label='KO'),
                       Line2D([0], [0], marker='x', color='r', markersize=6, label='Drug')]
    for g in range(NG):
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[g], markersize=6, label=f'Group {g}'))
    ax.legend(handles=legend_elements, fontsize=7, ncol=2)

    ax.set_title(f'Epoch {epoch}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    data = create_toy_data()
    m_feats, m_lbl = data['m_feats'].to(device), data['m_lbl'].to(device)
    c_feats, c_lbl = data['c_feats'].to(device), data['c_lbl'].to(device)
    d_feats, d_lbl = data['d_feats'].to(device), data['d_lbl'].to(device)

    model = CAAProjection(in_features=HD, embed_dim=args.embed_dim).to(device)

    # CE heads (non-ctrl classes only)
    m_ce = sorted(set(range(len(data['m_classes']))) - set(data['m_ctrl_group'].get('A', [])))
    d_ce = sorted(set(range(len(data['d_classes']))) - set(data['d_ctrl_group'].get('B', [])))
    ce_m = nn.Linear(args.embed_dim, len(m_ce)).to(device)
    ce_d = nn.Linear(args.embed_dim, len(d_ce)).to(device)
    m_ce_map = torch.zeros(len(data['m_classes']), dtype=torch.long, device=device)
    for n, o in enumerate(m_ce): m_ce_map[o] = n
    d_ce_map = torch.zeros(len(data['d_classes']), dtype=torch.long, device=device)
    for n, o in enumerate(d_ce): d_ce_map[o] = n

    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': list(ce_m.parameters()) + list(ce_d.parameters()), 'lr': args.lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    m_ctrl_t = torch.tensor(data['m_ctrl_group'].get('A', []), device=device)
    c_ctrl_A_t = torch.tensor(data['c_ctrl_group'].get('A', []), device=device)
    c_ctrl_B_t = torch.tensor(data['c_ctrl_group'].get('B', []), device=device)
    d_ctrl_t = torch.tensor(data['d_ctrl_group'].get('B', []), device=device)
    supcon = SupConLoss(temperature=args.tau)

    m_ds = TensorDataset(m_feats, m_lbl)
    c_ds = TensorDataset(c_feats, c_lbl)
    d_ds = TensorDataset(d_feats, d_lbl)
    n_nc = NG * NPG
    steps = n_nc * args.n_per_class // args.batch_size

    print(f"\n{'='*54}")
    print(f"Toy CAA — 3 groups×{NPG} classes (KO o | Drug x)  Steps/ep: {steps}")
    print(f"  lr={args.lr}  batch={args.batch_size}  epochs={args.epochs}")
    print(f"  λ_gw={args.lambda_gw}  λ_struct={args.lambda_struct}")
    print(f"  λ_unif={args.lambda_unif}  λ_ce={args.lambda_ce}")
    print(f"{'='*54}\n")

    model.eval()
    with torch.no_grad():
        z0 = model(torch.cat([m_feats, d_feats], dim=0))
    plot_tsne(z0[:len(m_feats)], m_lbl, d_lbl, m_ctrl_t, d_ctrl_t, 'init',
              os.path.join(OUTPUT_DIR, 'epoch_init.png'))

    ema_ancA = None
    ema_ancB = None

    for epoch in range(args.epochs):
        model.train()
        ce_m.train(); ce_d.train()
        t0 = time.time()

        ml = DataLoader(m_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        cl = DataLoader(c_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dl = DataLoader(d_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        mi, ci, di = iter(ml), iter(cl), iter(dl)

        rl = rg = rs_m = rs_d = ru = rc = 0.0

        for _ in range(steps):
            def _next(it, dl_):
                try: return next(it)
                except StopIteration: it = iter(dl_); return next(it)

            mx, mlb = _next(mi, ml); cx, clb = _next(ci, cl); dx, dlb = _next(di, dl)
            mx, mlb = mx.to(device), mlb.to(device)
            cx, clb = cx.to(device), clb.to(device)
            dx, dlb = dx.to(device), dlb.to(device)

            optimizer.zero_grad()
            zm = model(mx); zc = model(cx); zd = model(dx)

            def uc(ema, nz):
                if not len(nz): return ema
                c = F.normalize(nz.mean(0, keepdim=True), dim=1).squeeze(0)
                return c.detach() if ema is None else F.normalize(0.95 * ema + 0.05 * c.detach(), dim=0)

            za = []
            if len(m_ctrl_t): za.append(zm[torch.isin(mlb, m_ctrl_t)])
            if len(c_ctrl_A_t): za.append(zc[torch.isin(clb, c_ctrl_A_t)])
            if za: ema_ancA = uc(ema_ancA, torch.cat(za))

            zb = []
            if len(c_ctrl_B_t): zb.append(zc[torch.isin(clb, c_ctrl_B_t)])
            if len(d_ctrl_t): zb.append(zd[torch.isin(dlb, d_ctrl_t)])
            if zb: ema_ancB = uc(ema_ancB, torch.cat(zb))

            zm_nc = zm[~torch.isin(mlb, m_ctrl_t)] if len(m_ctrl_t) else zm
            zd_nc = zd[~torch.isin(dlb, d_ctrl_t)] if len(d_ctrl_t) else zd
            lm_nc = mlb[~torch.isin(mlb, m_ctrl_t)] if len(m_ctrl_t) else mlb
            ld_nc = dlb[~torch.isin(dlb, d_ctrl_t)] if len(d_ctrl_t) else dlb

            Lsm = supcon(zm_nc.unsqueeze(1), lm_nc) if len(zm_nc) > 1 else torch.tensor(0.0, device=device)
            Lsd = supcon(zd_nc.unsqueeze(1), ld_nc) if len(zd_nc) > 1 else torch.tensor(0.0, device=device)

            Lgw = torch.tensor(0.0, device=device)
            if ema_ancA is not None and ema_ancB is not None and len(zm_nc) and len(zd_nc):
                vm = F.normalize(zm_nc - ema_ancA.unsqueeze(0), dim=1)
                vd = F.normalize(zd_nc - ema_ancB.unsqueeze(0), dim=1)
                Lgw = gromov_wasserstein_loss(vm, vd, reg=0.05, outer_iter=10, inner_iter=20)

            Lun = uniformity_loss(torch.cat([zm_nc, zd_nc]), t=2.0) if len(zm_nc)+len(zd_nc) else torch.tensor(0.0, device=device)

            Lce = torch.tensor(0.0, device=device)
            if len(zm_nc): Lce += F.cross_entropy(ce_m(zm_nc), m_ce_map[lm_nc])
            if len(zd_nc): Lce += F.cross_entropy(ce_d(zd_nc), d_ce_map[ld_nc])

            loss = (args.lambda_gw * Lgw + args.lambda_struct * (Lsm + Lsd)
                    + args.lambda_unif * Lun + args.lambda_ce * Lce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            rl += loss.item(); rg += Lgw.item()
            rs_m += Lsm.item(); rs_d += Lsd.item()
            ru += Lun.item(); rc += Lce.item()

        scheduler.step()
        n_s = steps
        print(f"E{epoch:3d}: L={rl/n_s:.3f} GW={rg/n_s:.4f} CE={rc/n_s:.3f} "
              f"SupM={rs_m/n_s:.3f} SupD={rs_d/n_s:.3f} Unif={ru/n_s:.3f} [{time.time()-t0:.0f}s]")

        if (epoch + 1) % args.plot_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z = model(torch.cat([m_feats, d_feats], dim=0))
            plot_tsne(z[:len(m_feats)], m_lbl, d_lbl, m_ctrl_t, d_ctrl_t, epoch + 1,
                      os.path.join(OUTPUT_DIR, f'epoch{epoch+1}.png'))

    print(f"\nDone! Plots in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
