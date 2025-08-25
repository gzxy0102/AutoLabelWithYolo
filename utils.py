import colorsys

def generate_distinct_colors(n):
    """生成n个视觉上不同的颜色"""
    # 使用HSV颜色空间，均匀分布色相
    colors = []
    for i in range(n):
        # 色相从0到1变化，饱和度和明度固定为0.7
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        # 转换为0-255范围
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors
