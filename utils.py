import colorsys
import random
from typing import List, Tuple


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """生成n个视觉上不同的颜色
    
    使用HSV颜色空间，均匀分布色相，并添加一些变化以增加颜色多样性
    
    Args:
        n: 需要生成的颜色数量
    
    Returns:
        颜色列表，每个颜色为(r, g, b)元组
    """
    if n <= 0:
        return []
    
    # 预定义一些基础颜色，当n较小时优先使用
    if n <= 8:
        base_colors = [
            (255, 0, 0),     # 红色
            (0, 255, 0),     # 绿色
            (0, 0, 255),     # 蓝色
            (255, 255, 0),   # 黄色
            (255, 0, 255),   # 品红
            (0, 255, 255),   # 青色
            (255, 165, 0),   # 橙色
            (128, 0, 128)    # 紫色
        ]
        return base_colors[:n]
    
    # 对于较大的n，使用HSV颜色空间并添加一些随机变化
    colors = []
    for i in range(n):
        # 基本色相均匀分布
        hue = i / n
        # 添加小的随机变化以避免相邻颜色过于相似
        hue_variation = 0.05
        hue = (hue + random.uniform(-hue_variation, hue_variation)) % 1.0
        
        # 根据索引调整饱和度和明度，增加颜色多样性
        saturation = 0.6 + 0.2 * random.random()  # 0.6-0.8
        value = 0.8 + 0.1 * random.random()       # 0.8-0.9
        
        # 转换到RGB空间
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # 转换为0-255范围
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors


def generate_random_colors(n: int) -> List[Tuple[int, int, int]]:
    """生成n个随机颜色
    
    Args:
        n: 需要生成的颜色数量
    
    Returns:
        颜色列表，每个颜色为(r, g, b)元组
    """
    if n <= 0:
        return []
    
    # 使用列表推导式优化性能
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n)]


def ensure_directory(directory: str) -> None:
    """确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    import os
    os.makedirs(directory, exist_ok=True)