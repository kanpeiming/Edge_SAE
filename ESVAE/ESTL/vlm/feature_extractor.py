# -*- coding: utf-8 -*-
"""
VLM Feature Extractor
VLM特征提取器 - 支持CLIP/BLIP等模型

核心功能:
1. 提取全局图像特征
2. 提取空间特征图
3. 生成注意力图
4. 计算类别相关性
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from PIL import Image


class VLMFeatureExtractor:
    """
    VLM特征提取器
    
    支持的模型:
    - CLIP (openai/clip-vit-base-patch16, openai/clip-vit-base-patch32)
    - BLIP (Salesforce/blip-image-captioning-base)
    - SigLIP (google/siglip-base-patch16-224)
    
    推荐: CLIP-ViT-B/16 (平衡性能和速度)
    """
    
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch16",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cache_dir: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace模型名称
            device: 计算设备
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        
        # 加载模型和处理器
        self._load_model()
        
        print(f"✓ VLM模型加载成功: {model_name}")
        print(f"  设备: {device}")
    
    def _load_model(self):
        """加载VLM模型"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model.eval()
            self.model_type = 'clip'
            
        except Exception as e:
            raise RuntimeError(
                f"加载VLM模型失败: {e}\n"
                f"请确保已安装transformers: pip install transformers"
            )
    
    @torch.no_grad()
    def extract_features(self,
                        rgb_image: np.ndarray,
                        class_name: str,
                        return_attention: bool = True) -> Dict:
        """
        提取VLM特征
        
        Args:
            rgb_image: (H, W, 3) RGB图像 (numpy array, 0-255)
            class_name: 类别名称，如 "accordion", "airplanes"
            return_attention: 是否返回注意力图
            
        Returns:
            features: {
                'global_feat': (D,) 全局特征向量
                'spatial_feat': (H', W', D) 空间特征图
                'attention_map': (H, W) 注意力权重 [0, 1]
                'class_relevance': float 类别相关性分数
                'text_feat': (D,) 文本特征向量
            }
        """
        # 1. 准备输入
        # 转换为PIL Image
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = rgb_image
        
        # 构造文本提示
        text_prompt = self._create_text_prompt(class_name)
        
        # 处理输入
        inputs = self.processor(
            text=[text_prompt],
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 2. 前向传播
        outputs = self.model(**inputs, output_attentions=return_attention)
        
        # 3. 提取特征
        features = {}
        
        # 3.1 全局特征
        global_feat = outputs.image_embeds[0]  # (D,)
        features['global_feat'] = global_feat.cpu().numpy()
        
        # 3.2 文本特征
        text_feat = outputs.text_embeds[0]  # (D,)
        features['text_feat'] = text_feat.cpu().numpy()
        
        # 3.3 类别相关性（图像-文本相似度）
        similarity = (global_feat @ text_feat).item()
        features['class_relevance'] = similarity
        
        # 3.4 空间特征图
        vision_outputs = self.model.vision_model(
            pixel_values=inputs['pixel_values'],
            output_attentions=return_attention
        )
        
        # 从ViT的patch embeddings提取空间特征
        # last_hidden_state: (B, num_patches+1, D)
        # 去掉CLS token (第一个token)
        spatial_tokens = vision_outputs.last_hidden_state[0, 1:]  # (num_patches, D)
        
        # Reshape到空间维度
        num_patches = spatial_tokens.shape[0]
        patch_size = int(np.sqrt(num_patches))
        spatial_feat = spatial_tokens.reshape(patch_size, patch_size, -1)
        features['spatial_feat'] = spatial_feat.cpu().numpy()
        
        # 3.5 注意力图
        if return_attention:
            attention_map = self._compute_attention_map(
                vision_outputs.attentions,
                rgb_image.shape[:2] if isinstance(rgb_image, np.ndarray) else (224, 224)
            )
            features['attention_map'] = attention_map
        
        return features
    
    def _create_text_prompt(self, class_name: str) -> str:
        """
        创建文本提示
        
        策略:
        - 简单类别名: "a photo of a {class_name}"
        - 可以根据类别定制更具体的提示
        """
        # 处理类别名（去除下划线，转小写）
        class_name_clean = class_name.replace('_', ' ').lower()
        
        # 基础提示
        prompt = f"a photo of a {class_name_clean}"
        
        return prompt
    
    def _compute_attention_map(self,
                               attentions: Tuple[torch.Tensor],
                               target_size: Tuple[int, int]) -> np.ndarray:
        """
        从ViT注意力计算空间注意力图
        
        策略:
        1. 使用最后一层的注意力（高层语义）
        2. 聚合多头注意力
        3. 提取CLS token对patch的注意力
        4. 上采样到目标分辨率
        
        Args:
            attentions: ViT注意力 tuple of (B, num_heads, seq_len, seq_len)
            target_size: (H, W) 目标分辨率
            
        Returns:
            attention_map: (H, W) 注意力图 [0, 1]
        """
        # 取最后一层的注意力
        last_attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
        
        # 平均多头注意力
        avg_attn = last_attn.mean(dim=0)  # (seq_len, seq_len)
        
        # CLS token (第一个token) 对其他patch的注意力
        cls_attn = avg_attn[0, 1:]  # (num_patches,)
        
        # Reshape到空间维度
        num_patches = cls_attn.shape[0]
        patch_size = int(np.sqrt(num_patches))
        attn_map = cls_attn.reshape(patch_size, patch_size)
        
        # 上采样到目标分辨率
        attn_map = attn_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        attn_map_upsampled = F.interpolate(
            attn_map,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )[0, 0]  # (H, W)
        
        # 转为numpy并归一化
        attn_map_np = attn_map_upsampled.cpu().numpy()
        attn_map_normalized = self._normalize_array(attn_map_np)
        
        return attn_map_normalized
    
    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        """归一化数组到[0, 1]"""
        min_val = array.min()
        max_val = array.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        return ((array - min_val) / (max_val - min_val)).astype(np.float32)
    
    def batch_extract_features(self,
                               rgb_images: List[np.ndarray],
                               class_names: List[str],
                               batch_size: int = 8) -> List[Dict]:
        """
        批量提取特征
        
        Args:
            rgb_images: RGB图像列表
            class_names: 类别名称列表
            batch_size: 批次大小
            
        Returns:
            features_list: 特征字典列表
        """
        features_list = []
        
        num_images = len(rgb_images)
        for i in range(0, num_images, batch_size):
            batch_images = rgb_images[i:i+batch_size]
            batch_classes = class_names[i:i+batch_size]
            
            for img, cls in zip(batch_images, batch_classes):
                features = self.extract_features(img, cls)
                features_list.append(features)
        
        return features_list
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'device': self.device,
            'feature_dim': self.model.config.projection_dim,
            'image_size': self.processor.image_processor.size['shortest_edge'],
        }


# ============================================================================
# 辅助函数
# ============================================================================

def compare_vlm_models(rgb_image: np.ndarray,
                      class_name: str,
                      model_names: List[str] = None) -> Dict:
    """
    比较不同VLM模型的特征提取效果
    
    Args:
        rgb_image: RGB图像
        class_name: 类别名称
        model_names: 模型名称列表
        
    Returns:
        comparison: 比较结果字典
    """
    if model_names is None:
        model_names = [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
        ]
    
    comparison = {}
    
    for model_name in model_names:
        try:
            extractor = VLMFeatureExtractor(model_name=model_name)
            features = extractor.extract_features(rgb_image, class_name)
            
            comparison[model_name] = {
                'class_relevance': features['class_relevance'],
                'attention_map': features['attention_map'],
                'feature_dim': features['global_feat'].shape[0]
            }
            
            print(f"✓ {model_name}: relevance={features['class_relevance']:.4f}")
            
        except Exception as e:
            print(f"✗ {model_name}: {e}")
            comparison[model_name] = None
    
    return comparison

