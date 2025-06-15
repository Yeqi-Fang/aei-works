# generate_data.py

import numpy as np
import torch

def get_ground_truth_distribution_params(input_params):
    """
    一个“真理”函数，将10个输入参数映射到二维高斯混合模型的参数。
    这模拟了物理过程。
    
    输入:
        input_params (torch.Tensor): 形状为 (batch_size, 10) 的输入参数。
    
    输出:
        一个包含混合模型参数的字典。
    """
    # 使用一个简单的非线性变换来增加复杂性
    transformed = torch.sin(input_params[:, 0:1] * np.pi) + torch.cos(input_params[:, 1:2] * 0.5 * np.pi)
    
    params = {}
    
    # 混合权重 (让它依赖于输入参数)
    weight1 = torch.sigmoid(transformed) * 0.5 + 0.25  # 限制在 [0.25, 0.75]
    params['weight1'] = weight1
    params['weight2'] = 1.0 - weight1

    # 第一个高斯分量的均值
    mean1_x = 2 * torch.tanh(input_params[:, 2:3])
    mean1_y = 2 * torch.tanh(input_params[:, 3:4])
    params['mean1'] = torch.cat([mean1_x, mean1_y], dim=1)

    # 第二个高斯分量的均值
    mean2_x = -2 * torch.tanh(input_params[:, 4:5])
    mean2_y = -2 * torch.tanh(input_params[:, 5:6])
    params['mean2'] = torch.cat([mean2_x, mean2_y], dim=1)

    # 标准差 (必须为正)
    std1 = 0.5 * torch.sigmoid(input_params[:, 6:7]) + 0.1
    std2 = 0.5 * torch.sigmoid(input_params[:, 7:8]) + 0.1
    params['std1'] = torch.cat([std1, std1], dim=1) # 为简单起见，x和y的标准差相同
    params['std2'] = torch.cat([std2, std2], dim=1)

    return params

def sample_from_gmm(params, n_samples):
    """从高斯混合模型中采样"""
    n_batch = params['mean1'].shape[0]
    samples = []
    
    for i in range(n_batch):
        # 根据权重决定选择哪个分量
        choices = torch.multinomial(torch.tensor([params['weight1'][i], params['weight2'][i]]), n_samples, replacement=True)
        
        # 从第一个分量采样
        samples1 = torch.randn(n_samples, 2) * params['std1'][i] + params['mean1'][i]
        
        # 从第二个分量采样
        samples2 = torch.randn(n_samples, 2) * params['std2'][i] + params['mean2'][i]
        
        # 根据选择组合样本
        batch_samples = torch.where((choices == 0).unsqueeze(1), samples1, samples2)
        samples.append(batch_samples)
        
    return torch.stack(samples)

def main():
    print("正在生成数据集...")
    n_datasets = 8000  # 我们要生成多少个不同的输入条件
    n_samples_per_dist = 500  # 每个条件下采样多少个点
    
    # 1. 生成10个随机输入参数
    input_parameters = torch.rand(n_datasets, 10) * 2 - 1 # 范围在 [-1, 1]

    # 2. 获取每个输入对应的真实分布参数
    gmm_params = get_ground_truth_distribution_params(input_parameters)
    
    # 3. 从每个分布中采样
    # 注意：为了让NFs学习，我们将每个采样点与其对应的输入参数配对
    # 这意味着我们需要将输入参数重复 n_samples_per_dist 次
    
    output_samples = sample_from_gmm(gmm_params, n_samples_per_dist) # 形状: (n_datasets, n_samples, 2)
    
    # 重塑数据以进行训练： (n_datasets * n_samples, features)
    # 每个采样点都需要知道它的“条件”是什么
    inputs_repeated = input_parameters.repeat_interleave(n_samples_per_dist, dim=0)
    outputs_flattened = output_samples.reshape(-1, 2)

    print(f"输入数据形状: {inputs_repeated.shape}")   # (1,000,000, 10)
    print(f"输出数据形状: {outputs_flattened.shape}") # (1,000,000, 2)

    # 4. 保存到文件
    torch.save({
        'inputs': inputs_repeated,
        'outputs': outputs_flattened
    }, 'gmm_dataset.pth')
    
    print("数据集已保存到 gmm_dataset.pth")

if __name__ == '__main__':
    main()