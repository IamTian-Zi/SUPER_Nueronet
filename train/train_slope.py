"""
train_slope.py — SLOPE 模型训练脚本 (标准回归版)

核心逻辑:
    1. 使用最纯粹、最稳定的 MSE 损失函数，保证模型正常收敛。
    2. 仅在验证集上计算并打印 误删率 和 保留率，不把它们写进 Loss 里干扰训练。
    3. 只要验证集 MAE 最低，就保存模型。
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# 1. 数据集定义
# ============================================================

class SlopeDataset(Dataset):
    def __init__(self, data_dir, split='train', val_ratio=0.1, seed=42, max_samples=None):
        self.data_dir = data_dir
        files = sorted(glob.glob(os.path.join(data_dir, 'scene_*.npz')))

        if len(files) == 0:
            raise ValueError(f"目录 {data_dir} 里没有 .npz 文件，请先运行 data.py")

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(files))
        n_val = max(1, int(len(files) * val_ratio))
        val_idx = set(indices[:n_val].tolist())

        if split == 'train':
            self.files = [files[i] for i in range(len(files)) if i not in val_idx]
        else:
            self.files = [files[i] for i in range(len(files)) if i in val_idx]

        self.data = []
        total = 0
        for f in tqdm(self.files, desc=f"加载 {split} 集"):
            d = np.load(f)
            grids = d['grids']
            goals = d['goals']
            slope = d['slope']

            for i in range(len(grids)):
                self.data.append((
                    grids[i].astype(np.float32),
                    goals[i].astype(np.float32),
                    slope[i].astype(np.float32)
                ))
                total += 1
                if max_samples is not None and total >= max_samples:
                    break
            if max_samples is not None and total >= max_samples:
                break

        print(f"[{split}] 加载 {len(self.files)} 个文件，共 {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid, goal, slope = self.data[idx]
        return (
            torch.from_numpy(grid),
            torch.from_numpy(goal),
            torch.tensor(slope, dtype=torch.float32)
        )

# ============================================================
# 2. 模型定义 (SlopeNet)
# ============================================================

class SlopeNet(nn.Module):
    def __init__(self, grid_size=(8, 8, 4), in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm3d(32)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(32 + 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, grid, goal):
        x = F.relu(self.bn1(self.conv1(grid)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)

        x = torch.cat([x, goal], dim=-1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # 确保输出非负
        x = F.softplus(x)
        return x.squeeze(-1)

# ============================================================
# 3. 训练与验证函数
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, total_mae, n_batches = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for i, (grid, goal, slope) in enumerate(pbar):
        grid, goal, slope = grid.to(device), goal.to(device), slope.to(device)

        optimizer.zero_grad()
        pred = model(grid, goal)
        loss = criterion(pred, slope)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += torch.abs(pred - slope).mean().item()
        n_batches += 1

        if i % 10 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / n_batches, total_mae / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    all_pred, all_true = [], []

    for grid, goal, slope in tqdm(loader, desc="Val"):
        grid, goal, slope = grid.to(device), goal.to(device), slope.to(device)
        pred = model(grid, goal)
        loss = criterion(pred, slope)

        total_loss += loss.item()
        total_mae += torch.abs(pred - slope).mean().item()

        all_pred.append(pred.cpu().numpy())
        all_true.append(slope.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    # 计算指标供观察 (Thresh = 2.0)
    TARGET_THRESH = 2.0
    keep_mask = all_pred < TARGET_THRESH
    drop_mask = ~keep_mask

    keep_ratio = keep_mask.mean()

    # 误保留率：保留下来的节点中，真实距离 > 3.0 的占比
    if keep_mask.sum() > 0:
        false_keep = (all_true[keep_mask] > TARGET_THRESH * 1.5).mean()
    else:
        false_keep = 0.0

    # 误删率：删掉的节点中，真实距离 < 1.0 的占比
    if drop_mask.sum() > 0:
        false_drop = (all_true[drop_mask] < TARGET_THRESH * 0.5).mean()
    else:
        false_drop = 0.0

    metrics = {
        'keep_ratio': keep_ratio,
        'false_keep': false_keep,
        'false_drop': false_drop
    }

    return total_loss / len(loader), total_mae / len(loader), metrics

# ============================================================
# 4. 导出 TorchScript
# ============================================================

def export_torchscript(model, grid_size, save_path):
    model.eval()
    L, W, H = grid_size
    dummy_grid = torch.randn(1, 3, L, W, H)
    dummy_goal = torch.randn(1, 3)

    traced = torch.jit.trace(model, (dummy_grid, dummy_goal))
    traced.save(save_path)

    print(f"\n[导出] TorchScript 模型已保存至: {save_path}")
    loaded = torch.jit.load(save_path)
    out = loaded(dummy_grid, dummy_goal)
    print(f"[验证] 导出模型的输出形状: {out.shape}")

# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser('训练 SLOPE 模型 (标准回归版)')
    parser.add_argument('--data_dir', type=str, default='data/data/train')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    print(f"[设备] 使用: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = SlopeDataset(args.data_dir, split='train', max_samples=args.max_samples)
    val_ds = SlopeDataset(args.data_dir, split='val', max_samples=args.max_samples)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("错误: 数据集为空")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SlopeNet().to(device)
    print(f"[模型] 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 回归最稳的标准 MSE
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_mae = float('inf')
    print("=" * 70)
    print("开始训练 (目标: 最小化 MAE)")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_mae, metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        keep = metrics['keep_ratio']
        f_keep = metrics['false_keep']
        f_drop = metrics['false_drop']

        print(f"\n[Epoch {epoch}/{args.epochs}]")
        print(f"  Val MAE={val_mae:.4f} | 保留率: {keep:.1%} | 误保留: {f_keep:.2%} | 误删: {f_drop:.4%}")

        # 只要 MAE 最低，就保存（最符合回归任务逻辑）
        if val_mae < best_mae:
            best_mae = val_mae
            save_path = os.path.join(args.save_dir, 'slope_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_mae': val_mae,
                'metrics': metrics
            }, save_path)
            print(f"  ✓ 保存最佳模型 (MAE={val_mae:.4f})")

    print("\n" + "=" * 70)
    print("训练完成，正在加载最佳模型并导出...")
    print("=" * 70)

    best_model_path = os.path.join(args.save_dir, 'slope_best.pth')
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"最佳模型指标: MAE={best_ckpt['best_mae']:.4f}, 误删率={best_ckpt['metrics']['false_drop']:.4%}, 保留率={best_ckpt['metrics']['keep_ratio']:.1%}")

        export_path = os.path.join(args.save_dir, 'slope_model.pt')
        export_torchscript(model, grid_size=(8, 8, 4), save_path=export_path)
    else:
        print("警告: 没有找到保存的模型。")


if __name__ == '__main__':
    main()
