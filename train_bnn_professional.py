# train_bnn_professional.py
"""
训练一个 **Bayesian Neural Network (BNN)**，用于学习
p( y | x )，其中 x∈R^{10}，y∈R^{2}。

- 变分推断：
  每一层使用 Bayes by Back‑prop (mean‑field 高斯)；训练目标为
  **ELBO = NLL + KL/|D|**。
- 似然函数：条件各向异性高斯，网络直接输出
  μ(x) ∈ R^2  和  log σ(x) ∈ R^2。
- 预测阶段：多次采样权重，依分布采样 y，得到完整预测云，与真
  GMM 分布相比对可视化。

依赖：只需 PyTorch≥1.10。无需额外库。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Matplotlib 中文 & 坐标轴设置
import matplotlib as mpl
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False

# -------------- 数据生成函数（用于可视化对比） --------------
from generate_data import get_ground_truth_distribution_params, sample_from_gmm

# -------------- 可选：GPU --------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# =============================================================
#                1.  Bayesian Linear Layer
# =============================================================

class BayesianLinear(nn.Module):
    """线性层 w/ 变分权重  N(mu, softplus(rho))"""

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        # 变分参数
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))

        # 固定各向同性高斯先验 N(0, prior_sigma^2)
        self.prior_sigma = prior_sigma
        self.register_buffer("prior_log_sigma", torch.tensor(math.log(prior_sigma)))

    def _sample_param(self, mu, rho):
        sigma = F.softplus(rho)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, sigma

    def forward(self, x):
        # 抽样权重
        weight, weight_sigma = self._sample_param(self.weight_mu, self.weight_rho)
        bias, bias_sigma = self._sample_param(self.bias_mu, self.bias_rho)

        # kl(w) + kl(b)
        kl_w = self._kl_divergence(weight, weight_sigma)
        kl_b = self._kl_divergence(bias, bias_sigma)
        self.kl = kl_w + kl_b  # 保存供外部读取

        return F.linear(x, weight, bias)

    def _kl_divergence(self, sample, sigma):
        # KL q(w|mu,sigma) || p(w|0,prior_sigma)
        prior_sigma = self.prior_sigma
        kl = (sigma.log() - math.log(prior_sigma) + (prior_sigma ** 2 + sample.pow(2)) / (2 * sigma.pow(2)) - 0.5)
        return kl.sum()

# =============================================================
#                2.  BNN 回归网络
# =============================================================

class BNNRegressor(nn.Module):
    """简单 3 层感知机，每层贝叶斯线性 + ReLU，最后输出 μ 和 log σ"""

    def __init__(self, in_dim=10, hidden_dim=128, out_dim=2):
        super().__init__()
        self.b1 = BayesianLinear(in_dim, hidden_dim)
        self.b2 = BayesianLinear(hidden_dim, hidden_dim)
        self.b3 = BayesianLinear(hidden_dim, 2 * out_dim)  # 输出 (μ, log σ)
        self.layers = [self.b1, self.b2, self.b3]

    def forward(self, x):
        kl_sum = 0.0
        h = F.relu(self.b1(x))
        kl_sum += self.b1.kl
        h = F.relu(self.b2(h))
        kl_sum += self.b2.kl
        out = self.b3(h)
        kl_sum += self.b3.kl
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu, log_sigma, kl_sum

# =============================================================
#                3.  负对数似然 & 训练循环
# =============================================================

@torch.no_grad()
def nll_gaussian(y, mu, log_sigma):
    return 0.5 * (((y - mu) / log_sigma.exp()).pow(2) + 2 * log_sigma).sum(dim=-1)


def train_bnn(model, train_loader, val_loader, epochs=20, kl_weight=1.0):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            mu, log_sigma, kl = model(x)
            nll = nll_gaussian(y, mu, log_sigma).mean()
            loss = nll + kl_weight * kl / len(train_loader.dataset)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                mu, log_sigma, kl = model(x)
                nll = nll_gaussian(y, mu, log_sigma).mean()
                loss = nll + kl_weight * kl / len(train_loader.dataset)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch:>2}/{epochs}]  训练损失: {train_loss:.4f}  |  验证损失: {val_loss:.4f}")

# =============================================================
#                4.  预测 + 可视化与指标
# =============================================================

@torch.no_grad()
def sample_bnn(model, x_cond, n_samples=1):
    """对给定条件，采样权重 + 输出 → 返回 y 样本 (n_samples, 2)"""
    x_rep = x_cond.repeat(n_samples, 1).to(DEVICE)
    mu, log_sigma, _ = model(x_rep)
    eps = torch.randn_like(mu)
    y = mu + eps * log_sigma.exp()
    return y.cpu().numpy()


def compare_distribution(model, test_condition):
    n_plot = 5000

    pred_samples = sample_bnn(model, test_condition, n_plot)
    gt_params = get_ground_truth_distribution_params(test_condition)
    gt_samples = sample_from_gmm(gt_params, n_plot).squeeze().numpy()

    # --- 画图 ---
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(gt_samples[:, 0], gt_samples[:, 1], s=8, alpha=0.5, c='blue')
    ax1.set_title('真实分布')
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(pred_samples[:, 0], pred_samples[:, 1], s=8, alpha=0.5, c='red')
    ax2.set_title('BNN 预测分布')
    ax2.set_aspect('equal')

    ax3 = fig.add_subplot(1, 3, 3)
    sns.kdeplot(x=gt_samples[:, 0], y=gt_samples[:, 1], color='blue', levels=8, ax=ax3, label='真实')
    sns.kdeplot(x=pred_samples[:, 0], y=pred_samples[:, 1], color='red', levels=8, ax=ax3, label='预测', linestyles='--')
    ax3.legend()
    ax3.set_title('等高线对比')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    # --- 数值指标 ---
    true_mean, pred_mean = gt_samples.mean(0), pred_samples.mean(0)
    print("真实均值:", true_mean)
    print("预测均值:", pred_mean)
    print("|Δμ|:", np.abs(true_mean - pred_mean))

    # 近似 Wasserstein
    sub = 1000
    cost = cdist(gt_samples[:sub], pred_samples[:sub])
    row, col = linear_sum_assignment(cost)
    wass = cost[row, col].mean()
    print(f"近似 Wasserstein: {wass:.4f}")

# =============================================================
#                5.  主函数
# =============================================================

def main():
    print("加载数据…")
    data = torch.load('gmm_dataset.pth', map_location=DEVICE)
    dataset = TensorDataset(data['inputs'], data['outputs'])

    # 80/20 划分
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    print(f"训练集: {len(train_set)}  验证集: {len(val_set)}")

    batch = 4096
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, drop_last=False)

    # 构建与训练 BNN
    bnn = BNNRegressor()
    train_bnn(bnn, train_loader, val_loader, epochs=15, kl_weight=1.0)

    # 随机新条件可视化对比
    test_cond = torch.rand(1, 10) * 2 - 1
    print("测试条件:", test_cond.numpy())
    compare_distribution(bnn, test_cond)


if __name__ == '__main__':
    main()
