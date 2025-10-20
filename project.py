import os
import json
from utils import generate_distinct_colors


class Project:
    """项目类，管理项目信息和设置，包括标签颜色和缓存标注信息"""

    def __init__(self, name=None, path=None):
        # 基本项目信息
        self.name = name if name else "未命名项目"
        self.path = path  # 项目文件(.yap)路径

        # 目录配置
        self.image_dir = ""
        self.model_path = ""
        self.output_dir = ""

        # 标签配置
        self.class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
                            "train", "truck", "boat", "traffic light"]  # 默认标签
        self.class_colors = generate_distinct_colors(len(self.class_names))

        # 处理配置
        self.last_processed_index = 0  # 上次处理的图片索引，用于恢复

        # 标注数据缓存
        self.image_paths = []  # 图片路径列表
        self.processed_images = {}  # 存储处理过的图片 {路径: (原图, 标注)}

    @property
    def has_image_dir(self):
        """检查是否设置了图片目录"""
        return bool(self.image_dir)

    @property
    def has_model_path(self):
        """检查是否设置了模型路径"""
        return bool(self.model_path)

    @property
    def has_output_dir(self):
        """检查是否设置了输出目录"""
        return bool(self.output_dir)

    def save(self, path=None):
        """保存项目到文件，缓存标注信息"""
        if path:
            self.path = path
        if not self.path:
            return False

        # 辅助函数：将NumPy类型转换为Python原生类型
        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj

        # 只保存必要的信息，不保存原始图像数据
        data = {
            "name": self.name,
            "image_dir": self.image_dir,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "class_names": self.class_names,
            "class_colors": self.class_colors,
            "last_processed_index": self.last_processed_index,
            "image_paths": self.image_paths,
            # 只保存标注信息，不保存图像数据
            "annotations": convert_numpy_types({
                path: anns for path, (img, anns) in self.processed_images.items()
            })
        }

        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存项目失败: {e}")
            return False

    def load(self, path):
        """从文件加载项目，恢复缓存的标注信息"""
        self.path = path
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.name = data.get("name", "未命名项目")
            self.image_dir = data.get("image_dir", "")
            self.model_path = data.get("model_path", "")
            self.output_dir = data.get("output_dir", "")
            self.class_names = data.get("class_names", [])
            self.class_colors = data.get("class_colors", generate_distinct_colors(len(self.class_names)))
            self.last_processed_index = data.get("last_processed_index", 0)
            self.image_paths = data.get("image_paths", [])

            # 确保颜色数量与标签数量一致
            if len(self.class_colors) != len(self.class_names):
                self.class_colors = generate_distinct_colors(len(self.class_names))

            # 只加载标注信息，图像需要重新加载
            self.processed_images = {}
            annotations = data.get("annotations", {})
            for path, anns in annotations.items():
                # 检查图像文件是否存在
                if os.path.exists(path):
                    self.processed_images[path] = (None, anns)  # 图像为None，需要时再加载

            return True
        except Exception as e:
            print(f"加载项目失败: {e}")
            return False

    @property
    def processed_count(self):
        """获取已处理（有标注信息）的图片数量"""
        # 使用集合提高查找性能
        image_paths_set = set(self.image_paths)
        count = 0
        # 只检查当前项目中的图片路径
        for path in self.processed_images:
            if path in image_paths_set and self.has_annotations(path):
                count += 1
        return count

    @property
    def remaining_count(self):
        """获取剩余未处理的图片数量"""
        return self.total_count - self.last_processed_index

    @property
    def total_count(self):
        """获取图片总数"""
        return len(self.image_paths)

    @property
    def is_ready(self):
        """检查项目是否已准备好进行处理"""
        return self.has_image_dir and self.has_model_path and self.total_count > 0

    @property
    def progress(self):
        """获取处理进度 (0-100)"""
        if self.total_count == 0:
            return 0
        return int((self.last_processed_index / self.total_count) * 100)

    def has_annotations(self, image_path):
        """检查图片是否有标注信息"""
        if image_path in self.processed_images:
            _, annotations = self.processed_images[image_path]
            # 添加类型检查，确保annotations不是None
            return annotations is not None and len(annotations) > 0
        return False

    def get_image_name(self, image_path):
        """获取图片文件名"""
        return os.path.basename(image_path)
