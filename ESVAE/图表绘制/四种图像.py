import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# 图像加载与处理函数
def load_image(image_path, resize=(256, 256)):
    """加载图像并转换为灰度图"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(resize)
    image = np.array(image)
    return image


# Sobel 边缘检测
def sobel_edge_detection(image):
    """应用 Sobel 算法进行边缘检测"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel 算子计算水平和垂直边缘
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 合并两个方向的边缘
    sobel_edge = np.hypot(sobel_x, sobel_y)
    sobel_edge = np.uint8(np.clip(sobel_edge, 0, 255))

    return sobel_edge


# Canny 边缘检测
def canny_edge_detection(image):
    """应用 Canny 算法进行边缘检测"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 使用 Canny 边缘检测算法
    canny_edge = cv2.Canny(gray, 100, 200)

    return canny_edge


# 显示图像的函数
def show_images(sobel_edge, canny_edge, combined_edge):
    # 创建一个包含三张子图的画布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示 Sobel 边缘图
    axes[0].imshow(sobel_edge, cmap='gray')
    axes[0].set_title("Sobel Edge")
    axes[0].axis('off')

    # 显示 Canny 边缘图
    axes[1].imshow(canny_edge, cmap='gray')
    axes[1].set_title("Canny Edge")
    axes[1].axis('off')

    # 显示合并后的边缘图（按通道合并）
    axes[2].imshow(combined_edge)
    axes[2].set_title("Combined Edge")
    axes[2].axis('off')

    # 显示所有图像
    plt.show()


# 主函数
def main():
    # 设置图像路径
    image_path = 'path_to_image.png'  # 更改为 PNG 图像路径

    # 加载输入图像
    image = load_image(image_path)

    # 提取 Sobel 边缘图
    sobel_edge = sobel_edge_detection(image)

    # 提取 Canny 边缘图
    canny_edge = canny_edge_detection(image)

    # 合并 Sobel 和 Canny 边缘图，形成三通道图像
    combined_edge = np.dstack((sobel_edge, canny_edge, canny_edge))

    # 显示结果
    show_images(sobel_edge, canny_edge, combined_edge)


if __name__ == "__main__":
    main()
