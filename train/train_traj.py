import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np

class TrajDataset(Dataset):
    def __init__(self,npz_path):
        data=np.load(npz_path)
        self.starts=torch.from_numpy(data['starts'])
        self.goals = torch.from_numpy(data['goals'])
        self.init_paths = torch.from_numpy(data['init_paths'])
        self.opt_qs = torch.from_numpy(data['opt_qs'])
        self.opt_ts = torch.from_numpy(data['opt_ts'])

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        return self.starts[idx], self.goals[idx], self.init_paths[idx], self.opt_qs[idx], self.opt_ts[idx]


class TrajWarmStartNet(nn.Module):
    def __init__(self, max_waypoints=10):
        super().__init__()
        self.max_wp = max_waypoints

        # 1D 卷积提取 A* 路径的局部几何特征
        # 输入形状: (B, 3, max_wp)，3 代表 xyz
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 全连接层：融合起终点全局信息
        # 64(特征) * max_wp + 3(起点) + 3(终点)
        fc_in_dim = 64 * self.max_wp + 12

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 预测头
        self.q_head = nn.Linear(128, (self.max_wp - 1) * 3)  # 预测路点
        self.t_head = nn.Linear(128, self.max_wp - 1)  # 预测时间

    def forward(self, start, goal, init_path):
        """
        start: (B, 3)
        goal: (B, 3)
        init_path: (B, max_wp, 3)
        """
        # 转换维度以适应 Conv1d: (B, max_wp, 3) -> (B, 3, max_wp)
        x = init_path.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 展平
        x = x.view(x.size(0), -1)

        # 拼接起终点
        x = torch.cat([x, start, goal], dim=1)

        # 全连接
        x = self.fc(x)

        # 预测输出
        pred_q = self.q_head(x).view(-1, self.max_wp - 1, 3)  # (B, max_wp-1, 3)
        pred_t = F.softplus(self.t_head(x))  # (B, max_wp-1)，时间必须 > 0

        return pred_q, pred_t


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = TrajDataset("data/traj_dataset.npz")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TrajWarmStartNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for start, goal, init_path, gt_q, gt_t in dataloader:
            start, goal, init_path = start.to(device), goal.to(device), init_path.to(device)
            gt_q, gt_t = gt_q.to(device), gt_t.to(device)

            optimizer.zero_grad()
            pred_q, pred_t = model(start, goal, init_path)

            # 计算损失
            loss_q = F.mse_loss(pred_q, gt_q)
            loss_t = F.mse_loss(pred_t, gt_t)
            loss = loss_q + loss_t

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")

    # 导出 TorchScript 供 C++ 调用
    model.eval()
    dummy_start = torch.randn(1, 6).to(device)
    dummy_goal = torch.randn(1, 6).to(device)
    dummy_path = torch.randn(1, 10, 3).to(device)

    traced_script_module = torch.jit.trace(model, (dummy_start, dummy_goal, dummy_path))
    traced_script_module.save("traj_warmstart_model.pt")
    print("模型已导出为 traj_warmstart_model.pt")


if __name__ == "__main__":
    train()