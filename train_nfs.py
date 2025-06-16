# train_nfs_professional.py

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib as mpl
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
# 从我们第一个脚本中导入数据生成函数，用于最终的可视化对比
from generate_data import get_ground_truth_distribution_params, sample_from_gmm

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation

# --- 设备选择 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {device}")

def build_nfs_model(context_features=10, flow_features=2):
    """构建条件归一化流模型"""
    base_dist = StandardNormal(shape=[flow_features])
    num_transforms = 8
    transforms = []
    for _ in range(num_transforms):
        transforms.append(RandomPermutation(features=flow_features))
        transforms.append(MaskedAffineAutoregressiveTransform(
            features=flow_features,
            hidden_features=64,
            context_features=context_features
        ))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

def train_and_validate(model, train_loader, val_loader, epochs=10):
    """
    新的训练与验证循环
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    print("开始训练...")
    for epoch in range(epochs):
        # --- 训练部分 ---
        model.train()
        train_loss = 0
        for inputs, outputs in train_loader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            optimizer.zero_grad()
            loss = -model.log_prob(inputs=outputs, context=inputs).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 验证部分 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.to(device), outputs.to(device)
                loss = -model.log_prob(inputs=outputs, context=inputs).mean()
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

def predict_and_plot_comparison(model, test_condition):
    """
    改进的可视化函数：提供多种对比方式避免遮挡
    """
    model.eval()
    n_plot_samples = 5000

    # --- 1. 获取模型预测的样本 ---
    with torch.no_grad():
        predicted_samples = model.sample(
            1, context=test_condition.to(device).repeat(n_plot_samples, 1)
        ).squeeze().cpu().numpy()

    # --- 2. 获取对应的真实样本 ---
    ground_truth_params = get_ground_truth_distribution_params(test_condition)
    ground_truth_samples = sample_from_gmm(ground_truth_params, n_plot_samples).squeeze().numpy()

    # --- 3. 创建多种可视化方案 ---
    print("正在绘制改进的对比图...")
    fig = plt.figure(figsize=(20, 12))
    
    # 方案1: 并排显示两个分布
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(ground_truth_samples[:, 0], ground_truth_samples[:, 1], 
                alpha=0.6, s=8, c='blue', label="真实分布")
    ax1.set_title("真实分布")
    ax1.set_xlabel("参数 1")
    ax1.set_ylabel("参数 2")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('equal', adjustable='box')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(predicted_samples[:, 0], predicted_samples[:, 1], 
                alpha=0.6, s=8, c='red', label="预测分布")
    ax2.set_title("预测分布")
    ax2.set_xlabel("参数 1")
    ax2.set_ylabel("参数 2")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_aspect('equal', adjustable='box')
    
    # 方案2: 等高线对比（无填充，只有轮廓）
    ax3 = fig.add_subplot(2, 3, 3)
    sns.kdeplot(x=ground_truth_samples[:, 0], y=ground_truth_samples[:, 1], 
                color="blue", ax=ax3, label="真实分布", linewidths=2, levels=6)
    sns.kdeplot(x=predicted_samples[:, 0], y=predicted_samples[:, 1], 
                color="red", ax=ax3, label="预测分布", linewidths=2, levels=6)
    ax3.set_title("等高线对比")
    ax3.set_xlabel("参数 1")
    ax3.set_ylabel("参数 2")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_aspect('equal', adjustable='box')
    
    # 方案3: 透明度叠加散点图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(ground_truth_samples[:, 0], ground_truth_samples[:, 1], 
                alpha=0.4, s=6, c='blue', label="真实分布")
    ax4.scatter(predicted_samples[:, 0], predicted_samples[:, 1], 
                alpha=0.4, s=6, c='red', label="预测分布")
    ax4.set_title("透明度叠加散点图")
    ax4.set_xlabel("参数 1")
    ax4.set_ylabel("参数 2")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_aspect('equal', adjustable='box')
    
    # 方案4: 半透明填充等高线
    ax5 = fig.add_subplot(2, 3, 5)
    sns.kdeplot(x=ground_truth_samples[:, 0], y=ground_truth_samples[:, 1], 
                cmap="Blues", fill=True, alpha=0.5, thresh=0.05, ax=ax5)
    sns.kdeplot(x=predicted_samples[:, 0], y=predicted_samples[:, 1], 
                cmap="Reds", fill=True, alpha=0.5, thresh=0.05, ax=ax5)
    ax5.set_title("半透明填充等高线")
    ax5.set_xlabel("参数 1")
    ax5.set_ylabel("参数 2")
    ax5.grid(True, linestyle='--', alpha=0.6)
    ax5.set_aspect('equal', adjustable='box')
    
    # 方案5: 残差分析 (Q-Q Plot)
    ax6 = fig.add_subplot(2, 3, 6)
    percs = np.linspace(0, 100, 101)
    qn_pred_dim1 = np.percentile(predicted_samples[:, 0], percs)
    qn_true_dim1 = np.percentile(ground_truth_samples[:, 0], percs)
    qn_pred_dim2 = np.percentile(predicted_samples[:, 1], percs)
    qn_true_dim2 = np.percentile(ground_truth_samples[:, 1], percs)

    ax6.plot(qn_true_dim1, qn_pred_dim1, 'o', ms=3, label='参数 1 Q-Q', alpha=0.7)
    ax6.plot(qn_true_dim2, qn_pred_dim2, 'x', ms=3, label='参数 2 Q-Q', alpha=0.7)
    
    # 绘制完美的45度参考线
    lims = [
        min(min(qn_true_dim1), min(qn_true_dim2), min(qn_pred_dim1), min(qn_pred_dim2)),
        max(max(qn_true_dim1), max(qn_true_dim2), max(qn_pred_dim1), max(qn_pred_dim2)),
    ]
    ax6.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='完美匹配线')
    
    ax6.set_xlabel("真实分位数")
    ax6.set_ylabel("预测分位数")
    ax6.set_title("Q-Q Plot 残差分析")
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.6)
    ax6.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
    
    # --- 4. 输出数值对比指标 ---
    print("\n=== 分布对比指标 ===")
    
    # 计算均值和协方差
    true_mean = np.mean(ground_truth_samples, axis=0)
    pred_mean = np.mean(predicted_samples, axis=0)
    true_cov = np.cov(ground_truth_samples.T)
    pred_cov = np.cov(predicted_samples.T)
    
    print(f"真实分布均值: [{true_mean[0]:.3f}, {true_mean[1]:.3f}]")
    print(f"预测分布均值: [{pred_mean[0]:.3f}, {pred_mean[1]:.3f}]")
    print(f"均值差异: [{abs(true_mean[0]-pred_mean[0]):.3f}, {abs(true_mean[1]-pred_mean[1]):.3f}]")
    
    print(f"\n真实分布协方差:\n{true_cov}")
    print(f"\n预测分布协方差:\n{pred_cov}")
    
    # 计算Wasserstein距离（近似）
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    
    # 简化版本的Wasserstein距离估计
    n_subset = min(1000, len(ground_truth_samples), len(predicted_samples))
    true_subset = ground_truth_samples[:n_subset]
    pred_subset = predicted_samples[:n_subset]
    
    cost_matrix = cdist(true_subset, pred_subset, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    wasserstein_approx = cost_matrix[row_ind, col_ind].mean()
    
    print(f"\n近似Wasserstein距离: {wasserstein_approx:.4f}")



def main():
    # 1. 加载数据
    print("加载数据集...")
    data = torch.load('gmm_dataset.pth', map_location=device)
    full_dataset = TensorDataset(data['inputs'], data['outputs'])

    # --- 新增: 划分训练集和验证集 ---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    batch_size = 2048
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. 构建模型
    model = build_nfs_model()
    
    # 3. 训练和验证模型
    # 为了得到好的结果，你可能需要增加epochs的数量
    train_and_validate(model, train_loader, val_loader, epochs=10) 
    
    # 4. 推理和深度可视化
    # 创建一个新的、从未见过的条件进行预测
    test_condition = torch.rand(1, 10) * 2 - 1
    print(f"\n使用新的测试条件进行预测:\n{test_condition.numpy()}")
    
    predict_and_plot_comparison(model, test_condition)

if __name__ == '__main__':
    main()
    
    
    # number of parameters in the model
    # model = build_nfs_model()
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"模型参数总数: {num_params}")