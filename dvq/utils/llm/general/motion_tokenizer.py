import torch
import numpy as np
import argparse
import os
import sys

project_root = '../../../../'
sys.path.append(project_root)

from dvq.utils.define_device import define_device


class MotionTokenizer:
    """
    一个封装了训练好的VQ-VAE模型的动作分词器。
    """

    def __init__(self, model, data_root, vqvae_checkpoint):
        self.data_root = data_root
        self.device = define_device()

        # 1. 加载模型架构
        self.model = model

        # 3. 设置为评估模式
        self.model.eval()

        # 4. 加载用于标准化的均值和标准差
        try:
            self.mean = torch.from_numpy(np.load(os.path.join(self.data_root, 'Mean.npy'))).float().to(self.device)
            self.std = torch.from_numpy(np.load(os.path.join(self.data_root, 'Std.npy'))).float().to(self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"错误：在 {self.data_root} 目录下找不到 Mean.npy 或 Std.npy 文件。")


    def tokenize(self, motion_feature_vector: np.ndarray) -> np.ndarray:
        """
        将单个动作特征向量序列转换为离散的Token序列。

        Args:
            motion_feature_vector (np.ndarray): 形状为 (T, 148) 的动作特征向量。

        Returns:
            np.ndarray: 形状为 (T_quantized,) 的一维整数Token序列。
        """
        # 确保输入是 torch.Tensor 并且在正确的设备上
        motion_tensor = torch.from_numpy(motion_feature_vector).float().to(self.device)

        # 1. Z-Score Normalization (标准化)
        motion_normalized = (motion_tensor - self.mean) / self.std

        # 2. 添加一个批次维度 (Batch dimension) -> (1, T, 148)
        motion_batch = motion_normalized.unsqueeze(0)

        # 3. 使用模型的 encode 方法进行编码/分词
        with torch.no_grad():
            # 模型的encode方法返回的是一个 (1, T_quantized) 的张量
            tokens_tensor = self.model.encode(motion_batch)

        # 4. 移除批次维度并转换回 NumPy 数组
        for i in range(len(tokens_tensor)):
            tokens_tensor[i] = tokens_tensor[i].squeeze(0).cpu().numpy()

        return np.array(tokens_tensor)

    def __call__(self, motion_feature_vector: np.ndarray) -> np.ndarray:
        """让分词器实例可以直接调用"""
        return self.tokenize(motion_feature_vector)


def main():
    """
    一个演示如何使用 MotionTokenizer 的示例主程序。
    """
    parser = argparse.ArgumentParser(description="Motion Tokenizer Demo")

    # --- 关键参数 ---
    parser.add_argument('--vqvae_checkpoint', type=str, default="../../../model/vqvae/humanml3d_263.pt")
    parser.add_argument('--data_root', type=str, default="../../../data/humanml3d_263/")
    parser.add_argument('--motion_file', type=str, default="000000.npy")
    # 注意：quantizer相关的参数(如mu)在推理时通常不需要，因为我们不更新码本

    args = parser.parse_args()

    # 1. 初始化分词器
    tokenizer = MotionTokenizer(args, args.data_root)

    # 2. 加载一个动作文件
    try:
        motion_data = np.load(os.path.join(args.data_root, 'new_joint_vecs', args.motion_file))
        print(f"\n成功加载动作文件: {args.motion_file}")
        print(f"原始动作形状: {motion_data.shape}")
    except Exception as e:
        print(f"错误：无法加载动作文件 {args.motion_file}: {e}")
        return

    # 3. 执行分词
    motion_tokens = tokenizer.tokenize(motion_data)

    # 4. 打印结果
    print(f"\n分词后的Token序列 (前20个): {motion_tokens[:20]}")
    print(f"Token序列总长度: {len(motion_tokens)}")
    print(f"Token的数据类型: {motion_tokens.dtype}")
    print("\n分词成功！")


if __name__ == '__main__':
    main()
