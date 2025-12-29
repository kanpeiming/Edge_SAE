import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeExtractionModule(nn.Module):
    def __init__(self, device, in_channels=3):
        super(SobelEdgeExtractionModule, self).__init__()
        self.device = device

        # 定义 Sobel 核
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32)

        # 扩展维度并重复以匹配输入通道数
        self.sobel_kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1).to(device)
        self.sobel_kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1).to(device)

    def forward(self, x):
        # x 的形状应为 [N, C, H, W]
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        # 计算 Sobel 卷积
        edge_x = F.conv2d(x, self.sobel_kernel_x, padding=1, groups=x.size(1))
        edge_y = F.conv2d(x, self.sobel_kernel_y, padding=1, groups=x.size(1))

        # 计算梯度幅值
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)  # 添加小常数以避免除以零

        # 跨通道平均，得到单通道边缘图
        edges = torch.mean(edges, dim=1, keepdim=True)

        return edges


# 使用Canny算法进行边缘提取
class CannyEdgeDetectionModule(nn.Module):
    def __init__(self, device, in_channels=3, low_threshold=0.3, high_threshold=0.8):
        super(CannyEdgeDetectionModule, self).__init__()
        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.in_channels = in_channels

        # Sobel Kernels for gradient computation
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]], dtype=torch.float32).to(device)

        self.sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]], dtype=torch.float32).to(device)

        # Gaussian kernel for smoothing (example 5x5 kernel)
        self.gaussian_kernel = self.get_gaussian_kernel(5, 1.0).to(device)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        # Step 1: Convert RGB to grayscale (for simplicity, use weighted average)
        x_gray = self.rgb_to_grayscale(x)

        # Step 2: Apply Gaussian Blur
        x_blurred = self.apply_gaussian_blur(x_gray)

        # Step 3: Compute gradients using Sobel kernels
        grad_x = F.conv2d(x_blurred, self.sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1), padding=1)
        grad_y = F.conv2d(x_blurred, self.sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1), padding=1)

        # Step 4: Compute edge magnitude and direction
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_direction = torch.atan2(grad_y, grad_x)

        # Step 5: Non-Maximum Suppression (NMS)
        edges = self.non_maximum_suppression(edge_magnitude, edge_direction)

        # Step 6: Edge Tracking by Hysteresis
        edges = self.edge_tracking_by_hysteresis(edges)

        return edges

    def rgb_to_grayscale(self, x):
        """Convert RGB image to grayscale using weighted sum, or pass through if already grayscale."""
        if x.size(1) == 3:
            # Use the formula: Gray = 0.299 * R + 0.587 * G + 0.114 * B
            r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.unsqueeze(1)  # Add channel dimension (1 channel)
        elif x.size(1) == 1:
            return x  # Already grayscale
        else:
            raise ValueError(f"Expected input with 1 or 3 channels, but got {x.size(1)} channels.")

    def get_gaussian_kernel(self, size, sigma):
        """Generate a Gaussian kernel."""
        kernel = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        mean = size // 2
        sum_val = 0
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=self.device)  # Make sure sigma is a tensor

        for i in range(size):
            for j in range(size):
                # Ensure dist_sq is a tensor
                dist_sq = torch.tensor((i - mean) ** 2 + (j - mean) ** 2, dtype=torch.float32, device=self.device)
                kernel[i, j] = torch.exp(-dist_sq / (2 * sigma_tensor ** 2))  # Use tensor for the calculation
                sum_val += kernel[i, j]

        kernel /= sum_val  # Normalize the kernel
        return kernel

    def apply_gaussian_blur(self, x):
        """Apply Gaussian Blur to the image."""
        # Using the 5x5 Gaussian kernel (for simplicity)
        return F.conv2d(x, self.gaussian_kernel.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1), padding=2)

    def non_maximum_suppression(self, magnitude, direction):
        """Perform Non-Maximum Suppression (NMS)."""
        # Note: The implementation here is quite simplified; NMS requires more complex handling for edge directions.
        # This function would suppress non-maximum gradients.
        return magnitude

    def edge_tracking_by_hysteresis(self, edges):
        """Perform edge tracking by hysteresis."""
        # Simple thresholding example (for real-world application, this would be more sophisticated)
        edges = torch.where(edges > self.high_threshold, torch.ones_like(edges), torch.zeros_like(edges))
        return edges


class EnhancedCannyEdgeDetectionModule(nn.Module):
    def __init__(self, device, in_channels=3, low_threshold=0.2, high_threshold=0.6):
        super(EnhancedCannyEdgeDetectionModule, self).__init__()
        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.in_channels = in_channels

        # Sobel Kernels for gradient computation
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)

        # Gaussian kernel for smoothing (example 5x5 kernel)
        self.gaussian_kernel = self.get_gaussian_kernel(5, 1.0).to(device).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        # Step 1: Convert RGB to grayscale
        x_gray = self.rgb_to_grayscale(x)

        # Step 2: Apply Gaussian Blur
        x_blurred = self.apply_gaussian_blur(x_gray)

        # Step 3: Compute gradients using Sobel kernels
        grad_x = F.conv2d(x_blurred, self.sobel_kernel_x, padding=1, groups=1)
        grad_y = F.conv2d(x_blurred, self.sobel_kernel_y, padding=1, groups=1)

        # Step 4: Compute edge magnitude and direction
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_direction = torch.atan2(grad_y, grad_x)

        # Step 5: Non-Maximum Suppression (NMS)
        edges_nms = self.non_maximum_suppression(edge_magnitude, edge_direction)

        # Step 6: Edge Tracking by Hysteresis
        edges = self.edge_tracking_by_hysteresis(edges_nms)

        return edges

    def rgb_to_grayscale(self, x):
        """Convert RGB image to grayscale using weighted sum, or pass through if already grayscale."""
        if x.size(1) == 3:
            # Use the formula: Gray = 0.299 * R + 0.587 * G + 0.114 * B
            r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.unsqueeze(1)  # Add channel dimension (1 channel)
        elif x.size(1) == 1:
            return x  # Already grayscale
        else:
            raise ValueError(f"Expected input with 1 or 3 channels, but got {x.size(1)} channels.")

    def get_gaussian_kernel(self, size, sigma):
        """Generate a Gaussian kernel using vectorized operations."""
        ax = torch.arange(-size // 2 + 1., size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')  # 使用 'ij' 索引以匹配二维坐标
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / torch.sum(kernel)  # 归一化核
        return kernel

    def apply_gaussian_blur(self, x):
        """Apply Gaussian Blur to the image."""
        return F.conv2d(x, self.gaussian_kernel, padding=2)

    def non_maximum_suppression(self, magnitude, direction):
        """Perform Non-Maximum Suppression (NMS) using vectorized operations."""
        # 将方向从弧度转换为度数
        angle = direction * 180. / torch.pi
        angle[angle < 0] += 180

        B, C, H, W = magnitude.size()
        Z = torch.zeros_like(magnitude)

        # 四个方向：0, 45, 90, 135
        mask_0 = ((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180))
        mask_45 = (22.5 <= angle) & (angle < 67.5)
        mask_90 = (67.5 <= angle) & (angle < 112.5)
        mask_135 = (112.5 <= angle) & (angle < 157.5)

        # Shifted versions of magnitude
        mag_shifted_left = torch.nn.functional.pad(magnitude, (1, 1, 0, 0))[:, :, :, :-2]
        mag_shifted_right = torch.nn.functional.pad(magnitude, (1, 1, 0, 0))[:, :, :, 2:]
        mag_shifted_up = torch.nn.functional.pad(magnitude, (0, 0, 1, 1))[:, :, :-2, :]
        mag_shifted_down = torch.nn.functional.pad(magnitude, (0, 0, 1, 1))[:, :, 2:, :]
        mag_shifted_up_left = torch.nn.functional.pad(magnitude, (1, 1, 1, 1))[:, :, :-2, :-2]
        mag_shifted_up_right = torch.nn.functional.pad(magnitude, (1, 1, 1, 1))[:, :, :-2, 2:]
        mag_shifted_down_left = torch.nn.functional.pad(magnitude, (1, 1, 1, 1))[:, :, 2:, :-2]
        mag_shifted_down_right = torch.nn.functional.pad(magnitude, (1, 1, 1, 1))[:, :, 2:, 2:]

        # 0 degrees
        mask = mask_0
        Z[mask] = ((magnitude[mask] >= mag_shifted_left[mask]) &
                   (magnitude[mask] >= mag_shifted_right[mask])).float()

        # 45 degrees
        mask = mask_45
        Z[mask] = ((magnitude[mask] >= mag_shifted_down_left[mask]) &
                   (magnitude[mask] >= mag_shifted_up_right[mask])).float()

        # 90 degrees
        mask = mask_90
        Z[mask] = ((magnitude[mask] >= mag_shifted_up[mask]) &
                   (magnitude[mask] >= mag_shifted_down[mask])).float()

        # 135 degrees
        mask = mask_135
        Z[mask] = ((magnitude[mask] >= mag_shifted_up_left[mask]) &
                   (magnitude[mask] >= mag_shifted_down_right[mask])).float()

        return Z

    def edge_tracking_by_hysteresis(self, edges):
        """Perform edge tracking by hysteresis using dilation."""
        # Apply double threshold
        strong_edges = edges > self.high_threshold
        weak_edges = (edges >= self.low_threshold) & (edges <= self.high_threshold)

        # Initialize output with strong edges
        output = strong_edges.clone()

        # Define a 3x3 kernel for dilation
        kernel = torch.ones((1, 1, 3, 3), device=self.device)

        # Perform dilation on strong edges
        dilated_strong = torch.nn.functional.conv2d(strong_edges.float(), kernel, padding=1)
        dilated_strong = dilated_strong > 0

        # Connect weak edges to strong edges
        connected_weak = weak_edges & dilated_strong

        # Combine strong edges with connected weak edges
        output = strong_edges | connected_weak

        return output.float()


class MultiThresholdCannyEdgeModule(nn.Module):
    """
    多阈值Canny边缘检测模块 - 输出3通道
    
    通道设计：
    - Channel 0: 弱边缘 (低阈值) - 捕获更多细节
    - Channel 1: 中等边缘 (中阈值) - 平衡
    - Channel 2: 强边缘 (高阈值) - 主要结构
    
    优势：
    1. 多尺度边缘信息
    2. 与RGB 3通道结构对应
    3. 预训练时更自然
    """
    
    def __init__(self, device, in_channels=3):
        super(MultiThresholdCannyEdgeModule, self).__init__()
        self.device = device
        self.in_channels = in_channels
        
        # 三个不同阈值的Canny检测器
        # 弱边缘：捕获更多细节
        self.canny_weak = EnhancedCannyEdgeDetectionModule(
            device=device,
            in_channels=in_channels,
            low_threshold=0.05,   # 低阈值：更敏感
            high_threshold=0.15
        )
        
        # 中等边缘：平衡
        self.canny_medium = EnhancedCannyEdgeDetectionModule(
            device=device,
            in_channels=in_channels,
            low_threshold=0.15,   # 中等阈值
            high_threshold=0.30
        )
        
        # 强边缘：主要结构
        self.canny_strong = EnhancedCannyEdgeDetectionModule(
            device=device,
            in_channels=in_channels,
            low_threshold=0.25,   # 高阈值：只保留显著边缘
            high_threshold=0.50
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) - RGB图像
        
        Returns:
            edges: (B, 3, H, W) - 3通道边缘图
                - Channel 0: 弱边缘
                - Channel 1: 中等边缘
                - Channel 2: 强边缘
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")
        
        # 提取三个不同强度的边缘
        edges_weak = self.canny_weak(x)      # (B, 1, H, W)
        edges_medium = self.canny_medium(x)  # (B, 1, H, W)
        edges_strong = self.canny_strong(x)  # (B, 1, H, W)
        
        # 拼接为3通道
        edges = torch.cat([edges_weak, edges_medium, edges_strong], dim=1)  # (B, 3, H, W)
        
        return edges


class EventBridgeHead(nn.Module):
    """
    Event Statistics Prediction Head for Stage 1 Bridge Training
    
    功能：从RGB-edge的backbone表示中预测DVS的事件统计信息
    设计：轻量级1x1卷积或MLP
    
    输入：bottleneck特征 (N, T, 256) 或 features特征
    输出：event density map (N, T, H, W) 或 spatial activity mask
    """
    
    def __init__(self, input_dim=256, output_size=48, prediction_type='density'):
        """
        Args:
            input_dim: bottleneck特征维度（默认256）
            output_size: 输出空间尺寸（默认48，对应Caltech101）
            prediction_type: 'density' (密度图) 或 'activity' (活动掩码)
        """
        super(EventBridgeHead, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        self.prediction_type = prediction_type
        
        # 轻量级预测头：MLP + 上采样
        # bottleneck (256) -> 隐藏层 (128) -> 空间特征 (H*W)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_size * output_size),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, bottleneck_features):
        """
        Args:
            bottleneck_features: (N, T, 256) - bottleneck输出
        
        Returns:
            event_stats: (N, T, 1, H, W) - 事件统计图
        """
        N, T, C = bottleneck_features.shape
        
        # 逐时间步预测
        # (N, T, 256) -> (N*T, 256)
        features_flat = bottleneck_features.view(N * T, C)
        
        # 通过预测头
        # (N*T, 256) -> (N*T, H*W)
        pred_flat = self.predictor(features_flat)
        
        # 重塑为空间形式
        # (N*T, H*W) -> (N, T, 1, H, W)
        event_stats = pred_flat.view(N, T, 1, self.output_size, self.output_size)
        
        return event_stats


def compute_event_statistics(dvs_data, stat_type='density', target_size=None):
    """
    从DVS数据计算事件统计信息（ground truth）
    
    Args:
        dvs_data: (N, T, 2, H, W) - DVS事件帧
        stat_type: 'density' (事件密度) 或 'activity' (活动掩码)
        target_size: 目标空间尺寸 (H, W)，如果提供则resize
    
    Returns:
        event_stats: (N, T, 1, H_target, W_target) - 事件统计图
    """
    import torch.nn.functional as F
    
    # 简单策略：跨通道求和/平均作为事件活动强度
    # DVS 2通道 -> 单通道密度图
    if stat_type == 'density':
        # 事件密度：两通道的平均
        event_stats = dvs_data.mean(dim=2, keepdim=True)  # (N, T, 1, H, W)
    elif stat_type == 'activity':
        # 事件活动：二值化
        event_stats = (dvs_data.sum(dim=2, keepdim=True) > 0).float()  # (N, T, 1, H, W)
    else:
        raise ValueError(f"Unknown stat_type: {stat_type}")
    
    # 如果指定了target_size，则resize
    if target_size is not None:
        N, T, C, H, W = event_stats.shape
        # 将(N, T, 1, H, W)重塑为(N*T, 1, H, W)以便resize
        event_stats = event_stats.view(N * T, C, H, W)
        # Resize到目标尺寸
        event_stats = F.interpolate(event_stats, size=target_size, mode='bilinear', align_corners=False)
        # 重塑回(N, T, 1, H_target, W_target)
        event_stats = event_stats.view(N, T, C, target_size[0], target_size[1])
    
    return event_stats