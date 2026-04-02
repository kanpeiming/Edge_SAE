"""
DVS事件数据可视化模块
Visualization module for DVS event data

功能:
- 导出单个样本的RGB / Grayscale / Edge(Sobel) / DVS(累积事件帧) 四张PNG
- 支持两种Caltech101 DVS格式：
  1) 预处理 .pt（TLCaltech101做RGB↔DVS配对）
  2) 原始 .bin（N-Caltech101，按类别目录，例：.../Leopards/image_0135.bin）

使用方法:
python visualization/dvs_visualizer.py --class_name leopards --output_dir visualization_output
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
from pretrain.Edge import SobelEdgeExtractionModule
from torchvision import transforms
from PIL import Image


class DVSVisualizer:
    """DVS数据可视化器"""
    
    def __init__(self, device='cuda', img_size=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = int(img_size)
        
        # 初始化边缘提取器（只使用Sobel）
        self.sobel_extractor = SobelEdgeExtractionModule(device=self.device, in_channels=3)
        
        # RGB数据变换
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5429, 0.5263, 0.4994), (0.2422, 0.2392, 0.2406))
        ])
        
        # DVS数据变换
        self.dvs_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        # 反归一化参数（用于edge提取与RGB保存）
        self._rgb_mean = torch.tensor([0.5429, 0.5263, 0.4994]).view(3, 1, 1)
        self._rgb_std = torch.tensor([0.2422, 0.2392, 0.2406]).view(3, 1, 1)

    def _resolve_label_to_name(self, dataset, label: int) -> str:
        # TLCaltech101里 categories 是类别名列表
        if hasattr(dataset, "categories") and 0 <= label < len(dataset.categories):
            return str(dataset.categories[label])
        return str(label)

    def load_sample_by_class(self, class_name='leopards', dvs_root=None):
        """
        从Caltech101中按类别名加载一个样本（RGB与DVS来自同一个dataset索引返回的pair）
        
        Args:
            class_name: 类别名（不区分大小写），例如 'leopards'
            dvs_root: Caltech101 DVS根目录（可选，默认用DIR['Caltech101DVS']）
            
        Returns:
            rgb_img: RGB图像 (3, H, W)
            dvs_data: DVS数据 (T, 2, H, W)
            label: int
            label_name: str
        """
        dvs_root = dvs_root or DIR.get('Caltech101DVS')
        if dvs_root is None:
            raise ValueError("Caltech101DVS path is not configured (DIR['Caltech101DVS'] missing).")

        # 检查DVS数据结构：如果没有train/test子目录，说明是直接包含类别的原始格式
        # 这种情况下，我们需要手动处理，不使用TLCaltech101
        has_train_dir = os.path.exists(os.path.join(dvs_root, 'train'))
        
        if not has_train_dir:
            # 直接从原始N-Caltech101格式加载（类别目录下的.bin文件）
            return self._load_sample_from_raw_ncaltech(class_name, dvs_root)
        
        # 使用TLCaltech101加载预处理的.pt格式
        dataset = TLCaltech101(
            root=DIR['Caltech101'],
            dvs_root=dvs_root,
            train=True,
            transform=self.rgb_transform,
            dvs_transform=True,  # 触发TLCaltech101内部的dvs_trans
        )

        target_name = str(class_name).lower()
        for idx in range(len(dataset)):
            (rgb_img, dvs_data), label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            else:
                label = int(label)
            name = self._resolve_label_to_name(dataset, label)
            if name.lower() == target_name:
                return rgb_img, dvs_data, label, name

        raise ValueError(f"Class '{class_name}' not found in Caltech101 dataset.")
    
    def _load_sample_from_raw_ncaltech(self, class_name, dvs_root):
        """
        从原始N-Caltech101格式加载样本（类别目录下的.bin文件）
        
        Args:
            class_name: 类别名
            dvs_root: DVS根目录（直接包含类别文件夹）
            
        Returns:
            rgb_img: RGB图像 (3, H, W)
            dvs_data: DVS数据 (T, 2, H, W) - 转换为累积帧格式
            label: int (0，因为我们只加载单个类别)
            label_name: str
        """
        # 查找匹配的类别目录
        all_dirs = [d for d in os.listdir(dvs_root) 
                   if os.path.isdir(os.path.join(dvs_root, d))]
        
        target_name = str(class_name).lower()
        matched_dir = None
        for d in all_dirs:
            if d.lower() == target_name:
                matched_dir = d
                break
        
        if matched_dir is None:
            raise ValueError(f"Class '{class_name}' not found in DVS root: {dvs_root}")
        
        # 获取该类别下的第一个.bin文件
        class_path = os.path.join(dvs_root, matched_dir)
        bin_files = sorted([f for f in os.listdir(class_path) if f.endswith('.bin')])
        
        if not bin_files:
            raise ValueError(f"No .bin files found in {class_path}")
        
        bin_path = os.path.join(class_path, bin_files[0])
        
        # 加载RGB图像
        rgb_root = DIR.get('Caltech101')
        if not rgb_root:
            raise ValueError("Caltech101 RGB path not configured")
        
        rgb_path = self._find_rgb_for_bin(bin_path, rgb_root)
        pil = Image.open(rgb_path).convert("RGB")
        rgb_img = self.rgb_transform(pil)
        
        # 加载DVS数据并转换为tensor格式
        dvs_vis = self.load_dvs_from_ncaltech_bin(
            bin_path=bin_path,
            img_size=self.img_size,
            time_start_ratio=0.0,
            time_end_ratio=1.0,
            sensor_hw=(180, 240),
        )
        
        # 将累积的RGB格式(H,W,3)转换为事件帧格式(T,2,H,W)
        # 这里我们创建一个单帧的表示
        dvs_tensor = torch.zeros(1, 2, self.img_size, self.img_size)
        dvs_tensor[0, 0] = torch.from_numpy(dvs_vis[:, :, 2])  # 正极性（蓝色通道）
        dvs_tensor[0, 1] = torch.from_numpy(dvs_vis[:, :, 1])  # 负极性（绿色通道）
        
        return rgb_img, dvs_tensor, 0, matched_dir
    
    def rgb_to_grayscale(self, rgb_img):
        """
        将RGB图像转换为灰度图
        
        Args:
            rgb_img: RGB图像 (3, H, W) - 已归一化
            
        Returns:
            gray_img: 灰度图 (1, H, W)
        """
        # 反归一化到[0,1]
        rgb_denorm = rgb_img.cpu() * self._rgb_std + self._rgb_mean
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)
        
        # 使用标准权重转换为灰度图: 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * rgb_denorm[0] + 0.587 * rgb_denorm[1] + 0.114 * rgb_denorm[2]
        
        return gray.unsqueeze(0)  # (1, H, W)
    
    def extract_edge(self, rgb_img):
        """
        提取RGB图像的边缘（只使用Sobel）
        
        Args:
            rgb_img: RGB图像 (3, H, W) 或 (B, 3, H, W)
            
        Returns:
            edge_img: Sobel边缘图 (1, H, W)
        """
        if rgb_img.dim() == 3:
            rgb_img = rgb_img.unsqueeze(0)  # (1, 3, H, W)

        # rgb_img 当前是归一化后的tensor，这里先反归一化到[0,1]再做Sobel更合理
        rgb_denorm = rgb_img.cpu() * self._rgb_std + self._rgb_mean
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1).to(self.device)
        
        # 只使用Sobel边缘提取
        sobel_edge = self.sobel_extractor(rgb_denorm)  # (1, 1, H, W)
        
        return sobel_edge.squeeze(0).cpu()  # (1, H, W)

    def _find_rgb_for_bin(self, bin_path: str, rgb_root: str) -> str:
        """
        根据bin路径推断并寻找对应RGB文件：
        - bin: .../Caltech101/Leopards/image_0135.bin
        - rgb: <rgb_root>/Leopards/image_0135.jpg (或 png/jpeg)
        """
        if os.path.isdir(bin_path):
            raise ValueError(f"bin_path points to a directory, not a .bin file: {bin_path}")

        class_name = os.path.basename(os.path.dirname(bin_path))
        stem = os.path.splitext(os.path.basename(bin_path))[0]
        candidates = [
            os.path.join(rgb_root, class_name, stem + ext)
            for ext in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

        # fallback1：按“同类别内排序位置”做bin→rgb映射（更可能对齐同一序号样本）
        class_dir = os.path.join(rgb_root, class_name)
        bin_dir = os.path.dirname(bin_path)

        def _extract_num(name: str) -> int:
            base = os.path.splitext(os.path.basename(name))[0]
            digits = "".join([c for c in base if c.isdigit()])
            try:
                return int(digits) if digits else 0
            except Exception:
                return 0

        if os.path.isdir(class_dir) and os.path.isdir(bin_dir):
            rgb_files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            bin_files = [f for f in os.listdir(bin_dir) if f.lower().endswith(".bin")]
            rgb_files = sorted(rgb_files, key=_extract_num)
            bin_files = sorted(bin_files, key=_extract_num)

            if rgb_files and bin_files:
                bin_base = os.path.basename(bin_path)
                try:
                    pos = bin_files.index(bin_base)
                except ValueError:
                    # 若文件名不完全一致（极少），退化为按数字找最接近的
                    target_n = _extract_num(bin_base)
                    pos = int(np.argmin([abs(_extract_num(b) - target_n) for b in bin_files]))

                pos = min(max(pos, 0), len(rgb_files) - 1)
                return os.path.join(class_dir, rgb_files[pos])

            # fallback2：取该类别目录下第一张图
            if rgb_files:
                return os.path.join(class_dir, rgb_files[0])

        raise FileNotFoundError(
            f"RGB file not found for bin: {bin_path} under rgb_root={rgb_root}. "
            f"Tried same-stem match and class-order mapping."
        )

    def _resolve_bin_path(self, bin_path: str) -> str:
        """
        允许传入：
        - 具体 .bin 文件路径
        - 类别目录路径（自动选择其中一个 .bin）
        """
        bin_path = bin_path.strip()
        if os.path.isdir(bin_path):
            files = sorted([f for f in os.listdir(bin_path) if f.lower().endswith(".bin")])
            if not files:
                raise FileNotFoundError(f"No .bin files found under directory: {bin_path}")
            return os.path.join(bin_path, files[0])
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"bin_path does not exist: {bin_path}")
        if not bin_path.lower().endswith(".bin"):
            raise ValueError(f"bin_path must be a .bin file (or a directory containing .bin): {bin_path}")
        return bin_path

    def load_dvs_from_ncaltech_bin(
        self,
        bin_path: str,
        img_size: int = 128,
        time_start_ratio: float = 0.0,
        time_end_ratio: float = 1.0,
        sensor_hw=(180, 240),
    ) -> np.ndarray:
        """
        读取N-Caltech101的.bin事件文件，并生成累积可视化RGB图 (H,W,3)。
        time_start_ratio/time_end_ratio 用于控制“时间窗长点/短点”（取事件流的一段）。
        """
        img_size = int(img_size)
        H0, W0 = int(sensor_hw[0]), int(sensor_hw[1])

        raw = np.fromfile(bin_path, dtype=np.uint8)
        if raw.size < 5:
            return np.zeros((img_size, img_size, 3), dtype=np.float32)
        raw = raw.reshape(-1, 5)

        x = raw[:, 0].astype(np.int32)
        y = raw[:, 1].astype(np.int32)
        p = ((raw[:, 2] >> 7) & 0x01).astype(np.int32)
        ts = (((raw[:, 2] & 0x7F).astype(np.int32) << 16) | (raw[:, 3].astype(np.int32) << 8) | raw[:, 4].astype(np.int32))

        # 时间截取
        t_min = int(ts.min())
        t_max = int(ts.max())
        if t_max > t_min:
            start_t = t_min + int((t_max - t_min) * float(time_start_ratio))
            end_t = t_min + int((t_max - t_min) * float(time_end_ratio))
            mask_t = (ts >= start_t) & (ts <= end_t)
            x, y, p = x[mask_t], y[mask_t], p[mask_t]

        # 缩放到目标分辨率
        xs = (x.astype(np.float32) * (img_size / float(W0))).astype(np.int32)
        ys = (y.astype(np.float32) * (img_size / float(H0))).astype(np.int32)
        xs = np.clip(xs, 0, img_size - 1)
        ys = np.clip(ys, 0, img_size - 1)

        pos = np.zeros((img_size, img_size), dtype=np.float32)
        neg = np.zeros((img_size, img_size), dtype=np.float32)

        pos_mask = p == 1
        neg_mask = p == 0
        np.add.at(pos, (ys[pos_mask], xs[pos_mask]), 1.0)
        np.add.at(neg, (ys[neg_mask], xs[neg_mask]), 1.0)

        # 归一化（用百分位避免极值导致整体发黑）
        def norm_map(m):
            vmax = np.percentile(m, 99.5) if m.max() > 0 else 1.0
            return np.clip(m / (vmax + 1e-8), 0, 1)

        pos_n = norm_map(pos)
        neg_n = norm_map(neg)

        rgb = np.zeros((img_size, img_size, 3), dtype=np.float32)
        rgb[:, :, 2] = pos_n  # blue
        rgb[:, :, 1] = neg_n  # green
        return rgb

    def export_from_bin(self, bin_path: str, rgb_root: str, output_dir: str, time_end_ratio: float = 1.0):
        """
        给定一个 .bin 文件路径，导出四张图：RGB / Grayscale / Edge / DVS(累积)
        """
        os.makedirs(output_dir, exist_ok=True)
        bin_path = self._resolve_bin_path(bin_path)
        rgb_path = self._find_rgb_for_bin(bin_path, rgb_root)

        # RGB tensor（归一化，用于保存/edge）
        pil = Image.open(rgb_path).convert("RGB")
        rgb_tensor = self.rgb_transform(pil)

        class_name = os.path.basename(os.path.dirname(bin_path))
        stem = os.path.splitext(os.path.basename(bin_path))[0]
        tag = f"{class_name}_{stem}"

        out_rgb = os.path.join(output_dir, f"{tag}_rgb.png")
        out_gray = os.path.join(output_dir, f"{tag}_gray.png")
        out_edge = os.path.join(output_dir, f"{tag}_edge.png")
        out_dvs = os.path.join(output_dir, f"{tag}_dvs.png")

        # RGB保存
        self.visualize_rgb_image(rgb_tensor, save_path=out_rgb)

        # Grayscale保存
        gray = self.rgb_to_grayscale(rgb_tensor)
        self.visualize_grayscale_image(gray, save_path=out_gray)

        # Edge保存
        edge = self.extract_edge(rgb_tensor)
        self.visualize_edge_image(edge, save_path=out_edge)

        # DVS保存（累积、时间窗可控）
        dvs_vis = self.load_dvs_from_ncaltech_bin(
            bin_path=bin_path,
            img_size=self.img_size,
            time_start_ratio=0.0,
            time_end_ratio=float(time_end_ratio),
            sensor_hw=(180, 240),
        )
        plt.figure(figsize=(6, 6))
        plt.imshow(dvs_vis)
        plt.title(f"DVS Accumulated ({tag})", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dvs, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved:\n  - {out_rgb}\n  - {out_gray}\n  - {out_edge}\n  - {out_dvs}")

    def dvs_to_rgb_accumulated(self, dvs_data: torch.Tensor) -> np.ndarray:
        """
        将DVS (T,2,H,W) 累积成一张RGB图用于保存：
        - 正极性 -> 蓝色通道
        - 负极性 -> 绿色通道
        """
        if not isinstance(dvs_data, torch.Tensor):
            dvs_data = torch.tensor(dvs_data)
        if dvs_data.dim() != 4 or dvs_data.shape[1] != 2:
            raise ValueError(f"Unexpected dvs_data shape: {tuple(dvs_data.shape)} (expect (T,2,H,W))")

        pos = dvs_data[:, 0].sum(dim=0).float().cpu().numpy()
        neg = dvs_data[:, 1].sum(dim=0).float().cpu().numpy()

        pos = pos / (pos.max() + 1e-8)
        neg = neg / (neg.max() + 1e-8)

        H, W = pos.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[:, :, 2] = pos  # blue
        rgb[:, :, 1] = neg  # green
        return rgb

    def save_rgb_edge_dvs(self, rgb_img, dvs_data, label_name: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        rgb_path = os.path.join(output_dir, f"{label_name}_rgb.png")
        gray_path = os.path.join(output_dir, f"{label_name}_gray.png")
        edge_path = os.path.join(output_dir, f"{label_name}_edge.png")
        dvs_path = os.path.join(output_dir, f"{label_name}_dvs.png")

        # RGB（反归一化后保存）
        self.visualize_rgb_image(rgb_img, save_path=rgb_path)

        # Grayscale
        gray = self.rgb_to_grayscale(rgb_img)
        self.visualize_grayscale_image(gray, save_path=gray_path)

        # Edge
        edge = self.extract_edge(rgb_img)
        self.visualize_edge_image(edge, save_path=edge_path)

        # DVS（累积事件帧）
        dvs_vis = self.dvs_to_rgb_accumulated(dvs_data)
        plt.figure(figsize=(6, 6))
        plt.imshow(dvs_vis)
        plt.title(f"DVS Accumulated ({label_name})", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(dvs_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved:\n  - {rgb_path}\n  - {gray_path}\n  - {edge_path}\n  - {dvs_path}")
    
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
        rgb_img_denorm = rgb_img * self._rgb_std + self._rgb_mean
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
    
    def visualize_grayscale_image(self, gray_img, save_path=None):
        """
        可视化灰度图像
        
        Args:
            gray_img: 灰度图 (1, H, W)
            save_path: 保存路径
        """
        plt.figure(figsize=(5, 5))
        plt.imshow(gray_img[0].numpy(), cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grayscale image saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_edge_image(self, edge_img, save_path=None):
        """
        可视化边缘图像（只使用Sobel）
        
        Args:
            edge_img: Sobel边缘图 (1, H, W)
            save_path: 保存路径
        """
        plt.figure(figsize=(6, 6))
        
        # 显示Sobel边缘
        plt.imshow(edge_img[0].numpy(), cmap='gray')
        plt.title('Sobel Edge Extraction', fontsize=14)
        plt.axis('off')
        
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
    parser = argparse.ArgumentParser(description='Export Caltech101 sample: RGB / Edge / DVS')
    parser.add_argument('--class_name', type=str, default='leopards',
                       help='Caltech101 class name to export (case-insensitive), e.g. leopards')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                       help='Output directory for png files')
    parser.add_argument('--dvs_bin', type=str, default='',
                       help='Path to a N-Caltech101 .bin file (e.g. .../Leopards/image_0135.bin). If set, export this exact sample.')
    parser.add_argument('--rgb_root', type=str, default='',
                       help='Caltech101 RGB root (101_ObjectCategories). If empty, use DIR[\"Caltech101\"].')
    parser.add_argument('--time_end_ratio', type=float, default=1.0,
                       help='Use only the first portion of the event stream (0~1]. Smaller -> shorter window; 1.0 -> full stream.')
    parser.add_argument('--dvs_root', type=str, default='',
                       help='Override Caltech101 DVS root (optional). Should contain train/ and test/ folders.')
    parser.add_argument('--img_size', type=int, default=128, help='Output image size (default: 128)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = DVSVisualizer(device=args.device, img_size=args.img_size)

    if args.dvs_bin.strip():
        rgb_root = args.rgb_root.strip() or DIR.get('Caltech101')
        if not rgb_root:
            raise ValueError("rgb_root is empty; please pass --rgb_root pointing to 101_ObjectCategories")
        visualizer.export_from_bin(
            bin_path=args.dvs_bin.strip(),
            rgb_root=rgb_root,
            output_dir=args.output_dir,
            time_end_ratio=args.time_end_ratio,
        )
    else:
        dvs_root = args.dvs_root.strip() or None
        rgb_img, dvs_data, label, label_name = visualizer.load_sample_by_class(
            class_name=args.class_name,
            dvs_root=dvs_root,
        )
        visualizer.save_rgb_edge_dvs(rgb_img, dvs_data, label_name=label_name, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

