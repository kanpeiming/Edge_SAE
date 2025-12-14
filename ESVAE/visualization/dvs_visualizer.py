"""
DVS事件数据可视化模块
Visualization module for DVS event data

功能:
- 可视化DVS事件流 (Event Stream) 的3D点云表示
- 可视化DVS事件帧 (Event Frame) 序列
- 可视化RGB图像
- 可视化RGB边缘图 (Sobel/Canny)
- 支持N-Caltech101和Caltech101数据集

使用方法:
python visualization/dvs_visualizer.py --dataset caltech101 --sample_idx 0 --time_steps 10
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataloader.caltech101 import TLCaltech101, DIR
from pretrain.Edge import SobelEdgeExtractionModule, CannyEdgeDetectionModule
from torchvision import transforms
from PIL import Image


class DVSVisualizer:
    """DVS数据可视化器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化边缘提取器
        self.sobel_extractor = SobelEdgeExtractionModule(device=self.device, in_channels=3)
        self.canny_extractor = CannyEdgeDetectionModule(device=self.device, in_channels=3)
        
        # RGB数据变换
        self.rgb_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5429, 0.5263, 0.4994), (0.2422, 0.2392, 0.2406))
        ])
        
        # DVS数据变换
        self.dvs_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
    
    def load_sample(self, dataset_name='caltech101', sample_idx=0):
        """
        加载一个样本数据
        
        Args:
            dataset_name: 数据集名称 ('caltech101')
            sample_idx: 样本索引
            
        Returns:
            rgb_img: RGB图像 (3, H, W)
            dvs_data: DVS数据 (T, 2, H, W)
            label: 标签
        """
        if dataset_name == 'caltech101':
            # 加载Caltech101数据
            dataset = TLCaltech101(
                root=DIR['Caltech101'],
                dvs_root=DIR['Caltech101DVS'],
                train=True,
                transform=self.rgb_transform,
                dvs_transform=self.dvs_transform
            )
            
            (rgb_img, dvs_data), label = dataset[sample_idx]
            
            return rgb_img, dvs_data, label
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def extract_edge(self, rgb_img):
        """
        提取RGB图像的边缘
        
        Args:
            rgb_img: RGB图像 (3, H, W) 或 (B, 3, H, W)
            
        Returns:
            edge_img: 边缘图 (2, H, W) - [Sobel, Canny]
        """
        if rgb_img.dim() == 3:
            rgb_img = rgb_img.unsqueeze(0)  # (1, 3, H, W)
        
        rgb_img = rgb_img.to(self.device)
        
        # 提取Sobel边缘
        sobel_edge = self.sobel_extractor(rgb_img)  # (1, 1, H, W)
        
        # 提取Canny边缘
        canny_edge = self.canny_extractor(rgb_img)  # (1, 1, H, W)
        
        # 合并为2通道
        edge_img = torch.cat([sobel_edge, canny_edge], dim=1)  # (1, 2, H, W)
        
        return edge_img.squeeze(0).cpu()  # (2, H, W)
    
    def visualize_event_stream_3d(self, dvs_data, save_path=None):
        """
        可视化DVS事件流的3D点云表示
        
        Args:
            dvs_data: DVS数据 (T, 2, H, W)
            save_path: 保存路径
        """
        T, C, H, W = dvs_data.shape
        
        # 创建3D图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取正负极性事件
        positive_events = []  # (t, x, y)
        negative_events = []
        
        for t in range(T):
            # 正极性 (通道0)
            pos_mask = dvs_data[t, 0] > 0.1
            pos_coords = torch.nonzero(pos_mask)
            if len(pos_coords) > 0:
                for coord in pos_coords:
                    positive_events.append([t, coord[1].item(), coord[0].item()])
            
            # 负极性 (通道1)
            neg_mask = dvs_data[t, 1] > 0.1
            neg_coords = torch.nonzero(neg_mask)
            if len(neg_coords) > 0:
                for coord in neg_coords:
                    negative_events.append([t, coord[1].item(), coord[0].item()])
        
        # 转换为numpy数组
        if positive_events:
            pos_events = np.array(positive_events)
            ax.scatter(pos_events[:, 1], pos_events[:, 2], pos_events[:, 0], 
                      c='blue', marker='.', s=1, alpha=0.6, label='Positive')
        
        if negative_events:
            neg_events = np.array(negative_events)
            ax.scatter(neg_events[:, 1], neg_events[:, 2], neg_events[:, 0], 
                      c='green', marker='.', s=1, alpha=0.6, label='Negative')
        
        ax.set_xlabel('X (width)')
        ax.set_ylabel('Y (height)')
        ax.set_zlabel('t (time)')
        ax.set_title('Event Stream (3D Point Cloud)')
        ax.legend()
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Event stream 3D saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_event_frames(self, dvs_data, frame_indices=[0, 4, 9], save_path=None):
        """
        可视化DVS事件帧序列
        
        Args:
            dvs_data: DVS数据 (T, 2, H, W)
            frame_indices: 要显示的帧索引列表
            save_path: 保存路径
        """
        T, C, H, W = dvs_data.shape
        n_frames = len(frame_indices)
        
        fig, axes = plt.subplots(1, n_frames, figsize=(4*n_frames, 4))
        if n_frames == 1:
            axes = [axes]
        
        for idx, frame_idx in enumerate(frame_indices):
            if frame_idx >= T:
                continue
            
            # 合并正负极性为一个图像 (正极性=蓝色, 负极性=绿色)
            frame = dvs_data[frame_idx]  # (2, H, W)
            
            # 创建RGB图像
            rgb_frame = np.zeros((H, W, 3))
            rgb_frame[:, :, 2] = frame[0].numpy()  # 蓝色通道 = 正极性
            rgb_frame[:, :, 1] = frame[1].numpy()  # 绿色通道 = 负极性
            
            axes[idx].imshow(rgb_frame)
            axes[idx].set_title(f'Frame $F_{{{frame_idx+1}}}$')
            axes[idx].axis('off')
        
        plt.suptitle('Event Frame Representation', fontsize=14, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Event frames saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_rgb_image(self, rgb_img, save_path=None):
        """
        可视化RGB图像
        
        Args:
            rgb_img: RGB图像 (3, H, W) - 已归一化
            save_path: 保存路径
        """
        # 反归一化
        mean = torch.tensor([0.5429, 0.5263, 0.4994]).view(3, 1, 1)
        std = torch.tensor([0.2422, 0.2392, 0.2406]).view(3, 1, 1)
        rgb_img_denorm = rgb_img * std + mean
        rgb_img_denorm = torch.clamp(rgb_img_denorm, 0, 1)
        
        # 转换为numpy (H, W, 3)
        img_np = rgb_img_denorm.permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.title('RGB Image')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RGB image saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_edge_image(self, edge_img, save_path=None):
        """
        可视化边缘图像
        
        Args:
            edge_img: 边缘图 (2, H, W) - [Sobel, Canny]
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sobel边缘
        axes[0].imshow(edge_img[0].numpy(), cmap='gray')
        axes[0].set_title('Sobel Edge')
        axes[0].axis('off')
        
        # Canny边缘
        axes[1].imshow(edge_img[1].numpy(), cmap='gray')
        axes[1].set_title('Canny Edge')
        axes[1].axis('off')
        
        # 合并边缘 (蓝色=Sobel, 绿色=Canny)
        H, W = edge_img.shape[1:]
        merged_edge = np.zeros((H, W, 3))
        merged_edge[:, :, 2] = edge_img[0].numpy()  # 蓝色
        merged_edge[:, :, 1] = edge_img[1].numpy()  # 绿色
        
        axes[2].imshow(merged_edge)
        axes[2].set_title('Merged Edge (2-channel)')
        axes[2].axis('off')
        
        plt.suptitle('RGB Edge Extraction', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Edge image saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_complete_sample(self, dataset_name='caltech101', sample_idx=0, 
                                  output_dir='visualization_output'):
        """
        完整可视化一个样本: RGB + Edge + DVS Event Stream + DVS Event Frames
        
        Args:
            dataset_name: 数据集名称
            sample_idx: 样本索引
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading sample {sample_idx} from {dataset_name}...")
        
        # 加载数据
        rgb_img, dvs_data, label = self.load_sample(dataset_name, sample_idx)
        
        print(f"Label: {label}")
        print(f"RGB shape: {rgb_img.shape}")
        print(f"DVS shape: {dvs_data.shape}")
        
        # 提取边缘
        print("Extracting edges...")
        edge_img = self.extract_edge(rgb_img)
        
        # 可视化RGB图像
        print("Visualizing RGB image...")
        self.visualize_rgb_image(
            rgb_img, 
            save_path=os.path.join(output_dir, f'sample_{sample_idx}_rgb.png')
        )
        
        # 可视化边缘图像
        print("Visualizing edge image...")
        self.visualize_edge_image(
            edge_img,
            save_path=os.path.join(output_dir, f'sample_{sample_idx}_edge.png')
        )
        
        # 可视化DVS事件流3D
        print("Visualizing DVS event stream (3D)...")
        self.visualize_event_stream_3d(
            dvs_data,
            save_path=os.path.join(output_dir, f'sample_{sample_idx}_event_stream_3d.png')
        )
        
        # 可视化DVS事件帧
        print("Visualizing DVS event frames...")
        T = dvs_data.shape[0]
        frame_indices = [0, T//2, T-1]  # 第1帧, 中间帧, 最后一帧
        self.visualize_event_frames(
            dvs_data,
            frame_indices=frame_indices,
            save_path=os.path.join(output_dir, f'sample_{sample_idx}_event_frames.png')
        )
        
        print(f"\nVisualization complete! Results saved to: {output_dir}/")
        print(f"  - RGB image: sample_{sample_idx}_rgb.png")
        print(f"  - Edge image: sample_{sample_idx}_edge.png")
        print(f"  - Event stream 3D: sample_{sample_idx}_event_stream_3d.png")
        print(f"  - Event frames: sample_{sample_idx}_event_frames.png")


def main():
    parser = argparse.ArgumentParser(description='DVS Event Data Visualization')
    parser.add_argument('--dataset', type=str, default='caltech101', 
                       choices=['caltech101'], help='Dataset name')
    parser.add_argument('--sample_idx', type=int, default=0, 
                       help='Sample index to visualize')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                       help='Output directory for visualization results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = DVSVisualizer(device=args.device)
    
    # 完整可视化
    visualizer.visualize_complete_sample(
        dataset_name=args.dataset,
        sample_idx=args.sample_idx,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

