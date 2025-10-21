import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from utils import generate_distinct_colors

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Project:
    """项目类，管理项目信息和设置，包括标签颜色和缓存标注信息"""

    def __init__(self, name: Optional[str] = None, path: Optional[str] = None):
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
        self.processed_images: Dict[str, Tuple[Optional[Any], List[Dict]]] = {}  # 存储处理过的图片 {路径: (原图, 标注)}
        
        # 添加已标注图片的快速查找集合
        self._labeled_images = set()

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

    def save(self, path: Optional[str] = None) -> bool:
        """保存项目到文件，缓存标注信息"""
        if path:
            self.path = path
        if not self.path:
            logger.error("无法保存项目：路径未设置")
            return False

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

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
            logger.info(f"项目保存成功: {self.path}")
            return True
        except Exception as e:
            logger.error(f"保存项目失败: {e}")
            return False

    def load(self, path: str) -> bool:
        """从文件加载项目，恢复缓存的标注信息"""
        if not os.path.exists(path):
            logger.error(f"无法加载项目：文件不存在 - {path}")
            return False
            
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

            # 重置数据结构
            self.processed_images = {}
            self._labeled_images = set()
            
            # 只加载标注信息，图像需要重新加载
            annotations = data.get("annotations", {})
            for path, anns in annotations.items():
                # 检查图像文件是否存在
                if os.path.exists(path):
                    self.processed_images[path] = (None, anns)  # 图像为None，需要时再加载
                    # 更新已标注图片集合
                    if anns and len(anns) > 0:
                        self._labeled_images.add(path)

            logger.info(f"项目加载成功: {self.path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"项目文件格式错误: {e}")
            return False
        except Exception as e:
            logger.error(f"加载项目失败: {e}")
            return False

    @property
    def processed_count(self) -> int:
        """获取已处理（有标注信息）的图片数量"""
        # 使用集合快速查找已标注的图片
        return len(self._labeled_images)
        
    def update_labeled_status(self, image_path: str, has_annotations: bool) -> None:
        """更新图片的标注状态"""
        if has_annotations:
            self._labeled_images.add(image_path)
        elif image_path in self._labeled_images:
            self._labeled_images.remove(image_path)

    @property
    def remaining_count(self) -> int:
        """获取剩余未处理的图片数量"""
        # 使用max确保不会出现负数
        return max(0, self.total_count - self.last_processed_index)
    
    def add_image_annotation(self, image_path: str, image: Optional[Any], annotations: List[Dict]) -> None:
        """添加或更新图片的标注信息"""
        self.processed_images[image_path] = (image, annotations)
        # 更新标注状态
        self.update_labeled_status(image_path, len(annotations) > 0)
    
    def remove_image_annotation(self, image_path: str) -> None:
        """移除图片的标注信息"""
        if image_path in self.processed_images:
            del self.processed_images[image_path]
            self.update_labeled_status(image_path, False)

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

    def has_annotations(self, image_path: str) -> bool:
        """检查图片是否有标注信息"""
        # 快速检查：先看是否在已标注集合中
        if image_path in self._labeled_images:
            return True
            
        if image_path in self.processed_images:
            _, annotations = self.processed_images[image_path]
            # 添加类型检查，确保annotations不是None
            has_annots = annotations is not None and len(annotations) > 0
            # 更新标注状态缓存
            self.update_labeled_status(image_path, has_annots)
            return has_annots
        return False

    def get_image_name(self, image_path):
        """获取图片文件名"""
        return os.path.basename(image_path)
