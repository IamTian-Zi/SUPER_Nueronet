import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from CNNh import CNNH

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[警告] 未安装 tensorboard，跳过可视化")

class CNNH_Dataset(Dataset):
    """
       从 .npz 文件加载训练数据。
       每个 .npz 包含:
           grids:  (M, 3, L, W, H) float16
           goals:  (M, 3)          float32  终点相对坐标
           cnn_h:  (M,)            float32  真实剩余代价
           slope:  (M,)            float32  SLOPE 标签（本模型不用）
       """
    def __init__(self,data_dir,split='train',val_ratio=0.1,seed=42,max_samples=None):
        self.data_dir=data_dir
        files = sorted(glob.glob(os.path.join(data_dir, 'scene_*.npz')))

        if len(files) == 0:
            raise ValueError(f"目录 {data_dir} 里没有 .npz 文件")

        # 划分 train/val
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(files))
        n_val = max(1, int(len(files) * val_ratio))
        val_idx = set(indices[:n_val].tolist())

        if split == 'train':
            self.files = [files[i] for i in range(len(files)) if i not in val_idx]
        else:
            self.files = [files[i] for i in range(len(files)) if i in val_idx]

        # 预加载数据
        self.data = []
        total = 0
        for f in tqdm(self.files, desc=f"加载 {split}"):
            d = np.load(f)
            grids = d['grids']  # (M, 3, L, W, H)
            goals = d['goals']  # (M, 3)
            cnn_h = d['cnn_h']  # (M,)

            for i in range(len(grids)):
                self.data.append((
                    grids[i].astype(np.float32),
                    goals[i].astype(np.float32),
                    cnn_h[i].astype(np.float32),
                ))
                total += 1
                if max_samples is not None and total >= max_samples:
                    break
            if max_samples is not None and total >= max_samples:
                break

        print(f"[{split}] 加载 {len(self.files)} 个文件，{len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid, goal, h_label = self.data[idx]

        # 计算欧氏距离（作为网络的输入参考）
        euclidean = np.linalg.norm(goal).astype(np.float32)

        return (
            torch.from_numpy(grid),  # (3, L, W, H)
            torch.from_numpy(goal),  # (3,)
            torch.tensor(euclidean, dtype=torch.float32),  # 标量
            torch.tensor(h_label, dtype=torch.float32),  # 标量
        )
def train_one_epoch(model, loader, optimizer, device, epoch,
                     writer=None, log_interval=50):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for i, (grid, goal, euclidean, h_label) in enumerate(pbar):
        grid = grid.to(device)
        goal = goal.to(device)
        euclidean = euclidean.to(device)
        h_label = h_label.to(device)

        optimizer.zero_grad()
        pred = model(grid, goal, euclidean)

        # 损失: MSE + 可采纳性惩罚
        mse = F.mse_loss(pred, h_label)
        # 惩罚预测值超过真实代价（破坏可采纳性）
        over_penalty = F.relu(pred - h_label).mean()
        loss = mse + 0.1 * over_penalty

        loss.backward()
        optimizer.step()

        mae = torch.abs(pred - h_label).mean()
        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae.item():.4f}',
        })

        if writer is not None and i % log_interval == 0:
            step = epoch * len(loader) + i
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/mae', mae.item(), step)

    return total_loss / n_batches, total_mae / n_batches

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_mae = 0.0
    n_batches = 0

    all_pred = []
    all_true = []
    all_eucl = []

    pbar = tqdm(loader, desc="Val")
    for grid, goal, euclidean, h_label in pbar:
        grid = grid.to(device)
        goal = goal.to(device)
        euclidean = euclidean.to(device)
        h_label = h_label.to(device)

        pred = model(grid, goal, euclidean)
        mae = torch.abs(pred - h_label).mean()

        total_mae += mae.item()
        n_batches += 1

        all_pred.append(pred.cpu().numpy())
        all_true.append(h_label.cpu().numpy())
        all_eucl.append(euclidean.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    all_eucl = np.concatenate(all_eucl)

    # 关键指标
    overestimate_rate = (all_pred > all_true * 1.1).mean()  # 高估率
    underestimate_rate = (all_pred < all_true * 0.9).mean()  # 低估率

    metrics = {
        'mae': total_mae / n_batches,
        'overestimate_rate': overestimate_rate,
        'underestimate_rate': underestimate_rate,
        'mean_pred': all_pred.mean(),
        'mean_true': all_true.mean(),
    }

    return metrics

# ============================================================
# 4. 导出 TorchScript
# ============================================================

def export_torchscript(model, grid_size, save_path):
    model.eval()
    L, W, H = grid_size

    dummy_grid = torch.randn(1, 3, L, W, H)
    dummy_goal = torch.randn(1, 3)
    dummy_eucl = torch.randn(1)

    traced = torch.jit.trace(model, (dummy_grid, dummy_goal, dummy_eucl))
    traced.save(save_path)
    print(f"[导出] TorchScript 已保存: {save_path}")

    loaded = torch.jit.load(save_path)
    out = loaded(dummy_grid, dummy_goal, dummy_eucl)
    print(f"[验证] 导出模型输出形状: {out.shape}")

# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser('训练 CNN-h 启发函数')
    parser.add_argument('--data_dir', type=str, default='data/data/train')
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--grid_size', type=str, default='8,8,4')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--export_only', action='store_true')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--log_dir', type=str, default='logs/cnn_h')
    args = parser.parse_args()

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[设备] {device}")

    grid_size = tuple(int(x) for x in args.grid_size.split(','))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 数据
    if args.val_dir:
        train_ds = CNNH_Dataset(args.data_dir, split='train',
                                 val_ratio=0.0,
                                 max_samples=args.max_samples)
        val_ds = CNNH_Dataset(args.val_dir, split='train',
                               val_ratio=0.0,
                               max_samples=args.max_samples)
    else:
        train_ds = CNNH_Dataset(args.data_dir, split='train',
                                 val_ratio=args.val_ratio,
                                 max_samples=args.max_samples)
        val_ds = CNNH_Dataset(args.data_dir, split='val',
                               val_ratio=args.val_ratio,
                               max_samples=args.max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)

    # 模型
    model = CNNH(grid_size=grid_size, in_channels=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量: {n_params:,}")

    if args.resume:
        print(f"[加载] 从 {args.resume} 恢复")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_score = ckpt.get('best_score', float('inf'))  # ✅ 改成 best_score
        print(f"[加载] 已训练到 epoch {start_epoch - 1}, best_score={best_score:.4f}")
    else:
        start_epoch = 0
        best_score = float('inf')  # 改成 best_score

    model = model.to(device)

    if args.export_only:
        if not args.resume:
            print("错误：--export_only 需要配合 --resume 指定模型")
            sys.exit(1)
        export_path = os.path.join(args.save_dir, 'cnn_h_model.pt')
        export_torchscript(model, grid_size, export_path)
        return

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # TensorBoard
    writer = None
    if HAS_TB:
        writer = SummaryWriter(args.log_dir)
        print(f"[TensorBoard] tensorboard --logdir={args.log_dir}")

    # 训练循环
    print("=" * 70)
    print(f"开始训练 CNN-h，共 {args.epochs} 个 epoch")
    print("=" * 70)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        print(f"  Train: loss={train_loss:.4f}, mae={train_mae:.4f}")
        print(f"  Val:   mae={val_metrics['mae']:.4f}")
        print(f"         高估率(>110%真实): {val_metrics['overestimate_rate']:.2%}")
        print(f"         低估率(<90%真实):  {val_metrics['underestimate_rate']:.2%}")
        print(f"         预测均值={val_metrics['mean_pred']:.2f}, "
              f"真实均值={val_metrics['mean_true']:.2f}")

        score = (
                val_metrics['mae'] * 0.4
                + val_metrics['underestimate_rate'] * 1.0
                + val_metrics['overestimate_rate'] * 0.5
                + abs(val_metrics['mean_pred'] - val_metrics['mean_true']) * 0.1
        )
        print(f"  Composite Score: {score:.4f}")

        if writer:
            writer.add_scalar('val/mae', val_metrics['mae'], epoch)
            writer.add_scalar('val/overestimate_rate', val_metrics['overestimate_rate'], epoch)
            writer.add_scalar('val/score', score, epoch)  # 记录得分曲线

        # 保存最佳（根据 Score 判断）
        if score < best_score:  # ✅ 改为比较 Score
            best_score = score  # ✅ 更新 best_score
            save_path = os.path.join(args.save_dir, 'cnn_h_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,  # ✅ 保存 score
                'best_mae': val_metrics['mae'],  # 顺便也把当时的 MAE 存进去方便查看
                'grid_size': grid_size,
            }, save_path)
            print(f"  ✓ 保存最佳模型: {save_path} (score={best_score:.4f}, mae={val_metrics['mae']:.4f})")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'cnn_h_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'grid_size': grid_size,
            }, save_path)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"训练完成，耗时 {elapsed/60:.2f} 分钟")
    print(f"最佳验证 SCORE: {best_score:.4f}")
    print("=" * 70)

    # 导出 TorchScript
    print("\n[导出] 从最佳模型导出 TorchScript...")
    best_ckpt = torch.load(os.path.join(args.save_dir, 'cnn_h_best.pth'),
                            map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    export_path = os.path.join(args.save_dir, 'cnn_h_model.pt')
    export_torchscript(model, grid_size, export_path)

    if writer:
        writer.close()


if __name__ == '__main__':
    main()