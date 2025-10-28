# -*- coding: utf-8 -*-
"""
Semantic Edge Generation with Multi-Scale Edge Detection
语义边缘生成 - 支持多尺度边缘检测

核心思想:
1. 结合Sobel(密集)和Canny(稀疏)的优势
2. 使用VLM注意力图进行语义加权
3. 支持多尺度边缘融合
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MultiScaleEdgeDetector:
    """
    多尺度边缘检测器
    
    策略:
    1. Sobel: 提供密集的梯度信息（贴近原图）
    2. Canny: 提供稀疏的强边缘（与DVS稀疏性匹配）
    3. 多尺度融合: 结合不同尺度的边缘信息
    
    针对DVS的优化:
    - DVS是稀疏的，所以Canny的稀疏性很匹配
    - 但Sobel提供更多的梯度细节，有助于学习
    - 混合策略: alpha * Canny + (1-alpha) * Sobel
    """
    
    def __init__(self, 
                 use_sobel: bool = True,
                 use_canny: bool = True,
                 use_laplacian: bool = False,
                 sobel_weight: float = 0.3,
                 canny_weight: float = 0.7,
                 canny_threshold1: int = 50,
                 canny_threshold2: int = 150,
                 blur_kernel_size: int = 5):
        """
        Args:
            use_sobel: 是否使用Sobel（密集边缘）
            use_canny: 是否使用Canny（稀疏边缘）
            use_laplacian: 是否使用Laplacian（二阶导数）
            sobel_weight: Sobel边缘权重（推荐0.3，因为DVS稀疏）
            canny_weight: Canny边缘权重（推荐0.7，匹配DVS稀疏性）
            canny_threshold1: Canny低阈值
            canny_threshold2: Canny高阈值
            blur_kernel_size: 高斯模糊核大小
        """
        self.use_sobel = use_sobel
        self.use_canny = use_canny
        self.use_laplacian = use_laplacian
        
        self.sobel_weight = sobel_weight
        self.canny_weight = canny_weight
        
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.blur_kernel_size = blur_kernel_size
        
        # 归一化权重
        total_weight = (
            (sobel_weight if use_sobel else 0) +
            (canny_weight if use_canny else 0)
        )
        if total_weight > 0:
            self.sobel_weight /= total_weight
            self.canny_weight /= total_weight
    
    def detect(self, rgb_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        多尺度边缘检测
        
        Args:
            rgb_image: (H, W, 3) RGB图像
            
        Returns:
            edges_dict: {
                'sobel': (H, W) Sobel边缘 [0, 1]
                'canny': (H, W) Canny边缘 [0, 1]
                'combined': (H, W) 融合边缘 [0, 1]
                'gradient_x': (H, W) X方向梯度
                'gradient_y': (H, W) Y方向梯度
            }
        """
        # 转灰度图
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_image
        
        # 高斯模糊去噪
        if self.blur_kernel_size > 0:
            gray_blurred = cv2.GaussianBlur(
                gray, 
                (self.blur_kernel_size, self.blur_kernel_size), 
                0
            )
        else:
            gray_blurred = gray
        
        edges_dict = {}
        combined_edges = np.zeros_like(gray, dtype=np.float32)
        
        # 1. Sobel边缘检测（密集，贴近原图）
        if self.use_sobel:
            sobel_x = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算梯度幅值
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_normalized = self._normalize(sobel_magnitude)
            
            edges_dict['sobel'] = sobel_normalized
            edges_dict['gradient_x'] = sobel_x
            edges_dict['gradient_y'] = sobel_y
            
            combined_edges += self.sobel_weight * sobel_normalized
        
        # 2. Canny边缘检测（稀疏，匹配DVS）
        if self.use_canny:
            canny_edges = cv2.Canny(
                gray_blurred, 
                self.canny_threshold1, 
                self.canny_threshold2
            )
            canny_normalized = canny_edges.astype(np.float32) / 255.0
            
            edges_dict['canny'] = canny_normalized
            combined_edges += self.canny_weight * canny_normalized
        
        # 3. Laplacian边缘检测（可选，二阶导数）
        if self.use_laplacian:
            laplacian = cv2.Laplacian(gray_blurred, cv2.CV_64F)
            laplacian_normalized = self._normalize(np.abs(laplacian))
            edges_dict['laplacian'] = laplacian_normalized
        
        # 融合边缘
        edges_dict['combined'] = np.clip(combined_edges, 0, 1)
        
        return edges_dict
    
    def _normalize(self, array: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]"""
        min_val = array.min()
        max_val = array.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        return ((array - min_val) / (max_val - min_val)).astype(np.float32)


class SemanticEdgeGenerator:
    """
    语义边缘生成器
    结合VLM注意力图和多尺度边缘检测
    
    工作流程:
    1. 多尺度边缘检测 (Sobel + Canny)
    2. VLM注意力加权
    3. 类别特定增强
    4. 后处理优化
    """
    
    def __init__(self,
                 edge_detector: Optional[MultiScaleEdgeDetector] = None,
                 use_vlm_weighting: bool = True,
                 use_class_enhancement: bool = True,
                 attention_power: float = 2.0,
                 min_edge_threshold: float = 0.1):
        """
        Args:
            edge_detector: 边缘检测器（None则使用默认配置）
            use_vlm_weighting: 是否使用VLM注意力加权
            use_class_enhancement: 是否使用类别特定增强
            attention_power: 注意力权重的幂次（>1增强对比度）
            min_edge_threshold: 最小边缘阈值（去除弱边缘）
        """
        if edge_detector is None:
            # 默认配置: 70% Canny (稀疏) + 30% Sobel (密集)
            # 匹配DVS的稀疏特性
            edge_detector = MultiScaleEdgeDetector(
                use_sobel=True,
                use_canny=True,
                sobel_weight=0.3,
                canny_weight=0.7,
                canny_threshold1=50,
                canny_threshold2=150
            )
        
        self.edge_detector = edge_detector
        self.use_vlm_weighting = use_vlm_weighting
        self.use_class_enhancement = use_class_enhancement
        self.attention_power = attention_power
        self.min_edge_threshold = min_edge_threshold
    
    def generate(self,
                 rgb_image: np.ndarray,
                 vlm_features: Optional[Dict] = None,
                 class_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        生成语义感知的边缘图
        
        Args:
            rgb_image: (H, W, 3) RGB图像
            vlm_features: VLM特征字典（包含attention_map等）
            class_name: 类别名称（用于类别特定增强）
            
        Returns:
            result_dict: {
                'semantic_edges': (H, W) 最终语义边缘 [0, 1]
                'raw_edges': (H, W) 原始边缘（未加权）
                'attention_map': (H, W) VLM注意力图
                'sobel_edges': (H, W) Sobel边缘
                'canny_edges': (H, W) Canny边缘
            }
        """
        # 1. 多尺度边缘检测
        edges_dict = self.edge_detector.detect(rgb_image)
        raw_edges = edges_dict['combined']
        
        result_dict = {
            'raw_edges': raw_edges,
            'sobel_edges': edges_dict.get('sobel', None),
            'canny_edges': edges_dict.get('canny', None),
        }
        
        semantic_edges = raw_edges.copy()
        
        # 2. VLM注意力加权
        if self.use_vlm_weighting and vlm_features is not None:
            attention_map = vlm_features.get('attention_map', None)
            
            if attention_map is not None:
                # 确保尺寸匹配
                if attention_map.shape != raw_edges.shape:
                    attention_map = cv2.resize(
                        attention_map,
                        (raw_edges.shape[1], raw_edges.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # 注意力加权（增强VLM关注的区域）
                # 使用幂次增强对比度
                attention_weighted = np.power(attention_map, self.attention_power)
                
                # 混合策略: 保留30%原始边缘 + 70%注意力加权边缘
                # 避免完全丢失VLM未关注但实际重要的边缘
                semantic_edges = (
                    0.3 * raw_edges + 
                    0.7 * raw_edges * attention_weighted
                )
                
                result_dict['attention_map'] = attention_map
        
        # 3. 类别特定增强（可选）
        if self.use_class_enhancement and vlm_features is not None:
            spatial_feat = vlm_features.get('spatial_feat', None)
            
            if spatial_feat is not None:
                enhanced_edges = self._class_specific_enhancement(
                    semantic_edges,
                    spatial_feat,
                    class_name
                )
                semantic_edges = enhanced_edges
        
        # 4. 后处理
        semantic_edges = self._post_process(semantic_edges)
        
        # 5. 阈值处理（去除弱边缘）
        semantic_edges[semantic_edges < self.min_edge_threshold] = 0
        
        result_dict['semantic_edges'] = semantic_edges
        
        return result_dict
    
    def _class_specific_enhancement(self,
                                    edges: np.ndarray,
                                    spatial_feat: np.ndarray,
                                    class_name: Optional[str] = None) -> np.ndarray:
        """
        类别特定的边缘增强
        利用VLM的空间特征图识别判别性区域
        
        Args:
            edges: (H, W) 边缘图
            spatial_feat: (H', W', C) VLM空间特征
            class_name: 类别名称
            
        Returns:
            enhanced_edges: (H, W) 增强后的边缘
        """
        # 计算空间特征的激活强度
        if len(spatial_feat.shape) == 3:
            activation = np.linalg.norm(spatial_feat, axis=-1)  # (H', W')
        else:
            activation = spatial_feat
        
        # 上采样到边缘图分辨率
        if activation.shape != edges.shape:
            activation = cv2.resize(
                activation,
                (edges.shape[1], edges.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # 归一化
        activation_norm = self._normalize_array(activation)
        
        # 增强判别性区域的边缘
        # 策略: 高激活区域的边缘权重增加
        enhanced_edges = edges * (0.5 + 0.5 * activation_norm)
        
        return enhanced_edges
    
    def _post_process(self, edges: np.ndarray) -> np.ndarray:
        """
        边缘后处理
        
        操作:
        1. 形态学闭运算（连接断裂边缘）
        2. 轻微高斯平滑（去除噪声）
        3. 归一化
        """
        # 1. 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_closed = cv2.morphologyEx(
            edges, 
            cv2.MORPH_CLOSE, 
            kernel,
            iterations=1
        )
        
        # 2. 轻微高斯平滑
        edges_smoothed = cv2.GaussianBlur(edges_closed, (3, 3), 0.5)
        
        # 3. 归一化
        edges_normalized = self._normalize_array(edges_smoothed)
        
        return edges_normalized
    
    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        """归一化数组到[0, 1]"""
        min_val = array.min()
        max_val = array.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        return ((array - min_val) / (max_val - min_val)).astype(np.float32)
    
    def batch_generate(self,
                       rgb_images: list,
                       vlm_features_list: Optional[list] = None,
                       class_names: Optional[list] = None) -> list:
        """
        批量生成语义边缘
        
        Args:
            rgb_images: RGB图像列表
            vlm_features_list: VLM特征列表
            class_names: 类别名称列表
            
        Returns:
            results: 结果字典列表
        """
        results = []
        
        for i, rgb_image in enumerate(rgb_images):
            vlm_feat = vlm_features_list[i] if vlm_features_list else None
            class_name = class_names[i] if class_names else None
            
            result = self.generate(rgb_image, vlm_feat, class_name)
            results.append(result)
        
        return results


# ============================================================================
# 辅助函数
# ============================================================================

def visualize_edges_comparison(rgb_image: np.ndarray,
                               edges_dict: Dict[str, np.ndarray],
                               save_path: Optional[str] = None) -> np.ndarray:
    """
    可视化不同边缘检测结果的对比
    
    Args:
        rgb_image: 原始RGB图像
        edges_dict: 边缘字典
        save_path: 保存路径（可选）
        
    Returns:
        comparison_image: 对比图像
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # Sobel
    if 'sobel_edges' in edges_dict and edges_dict['sobel_edges'] is not None:
        axes[0, 1].imshow(edges_dict['sobel_edges'], cmap='gray')
        axes[0, 1].set_title('Sobel (Dense)')
        axes[0, 1].axis('off')
    
    # Canny
    if 'canny_edges' in edges_dict and edges_dict['canny_edges'] is not None:
        axes[0, 2].imshow(edges_dict['canny_edges'], cmap='gray')
        axes[0, 2].set_title('Canny (Sparse)')
        axes[0, 2].axis('off')
    
    # Raw combined
    if 'raw_edges' in edges_dict:
        axes[1, 0].imshow(edges_dict['raw_edges'], cmap='gray')
        axes[1, 0].set_title('Combined Edges')
        axes[1, 0].axis('off')
    
    # VLM attention
    if 'attention_map' in edges_dict and edges_dict['attention_map'] is not None:
        axes[1, 1].imshow(edges_dict['attention_map'], cmap='hot')
        axes[1, 1].set_title('VLM Attention')
        axes[1, 1].axis('off')
    
    # Semantic edges
    if 'semantic_edges' in edges_dict:
        axes[1, 2].imshow(edges_dict['semantic_edges'], cmap='gray')
        axes[1, 2].set_title('Semantic Edges (Final)')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # 转换为numpy数组
    fig.canvas.draw()
    comparison_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison_image = comparison_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return comparison_image

