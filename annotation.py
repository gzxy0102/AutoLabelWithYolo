import random
import shutil

import cv2
import os

from PySide6.QtWidgets import QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox
from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal


class AnnotationThread(QThread):
    """标注线程，用于后台处理图片标注，支持暂停和继续"""
    progress_updated = Signal(int)
    image_processed = Signal(str, object, object)
    finished = Signal()

    def __init__(self, image_paths, model_path, class_names):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.class_names = class_names
        self.running = True
        self.paused = False
        try:
            self.model = YOLO(self.model_path)
            self.model.to(device="cuda")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.finished.emit()
            return

    def run(self):
        total = len(self.image_paths)
        for i, image_path in enumerate(self.image_paths):
            # 检查是否需要停止
            if not self.running:
                break
            # 检查是否需要暂停
            while self.paused and self.running:
                self.msleep(100)

            if not self.running:
                break

            try:
                # 处理图片（已在process_image中完成类型转换）
                image, annotations = self.process_image(image_path)

                # 发送处理完成的信号
                self.image_processed.emit(image_path, image, annotations)

            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")

            # 更新进度
            progress = int((i + 1) / total * 100)
            self.progress_updated.emit(progress)

        self.finished.emit()

    def process_image(self, image_path):
        """处理单张图片，返回转换后的标注数据"""
        try:
            # 加载图片
            image = cv2.imread(image_path)
            if image is None:
                return None, []

            # 模型预测（假设返回的是NumPy数组）
            results = self.model(image)
            annotations = []

            # 处理每个预测结果
            for result in results:
                boxes = result.boxes.cpu().numpy()
                model_names = result.names
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = model_names[cls]
                    if class_name in self.class_names:
                        annotations.append({
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": float(conf),
                            "class_id": int(cls),
                            "class": class_name
                        })

            return image, annotations
        except Exception as e:
            print(f"处理图片 {image_path} 失败: {e}")
            return None, []

    def pause(self):
        """暂停线程"""
        self.paused = True

    def resume(self):
        """继续线程"""
        self.paused = False

    def stop(self):
        """停止线程"""
        self.running = False
        self.paused = False
        self.wait()


class DatasetSplitDialog(QDialog):
    """数据集划分比例设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置数据集划分比例")

        # 默认比例 70%:15%:15%
        self.train_ratio = QLineEdit("70")
        self.val_ratio = QLineEdit("15")
        self.test_ratio = QLineEdit("15")

        layout = QFormLayout()
        layout.addRow("训练集比例 (%):", self.train_ratio)
        layout.addRow("验证集比例 (%):", self.val_ratio)
        layout.addRow("测试集比例 (%):", self.test_ratio)

        buttons = QDialogButtonBox()
        buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_ratios(self):
        """获取用户设置的比例，返回小数形式"""
        try:
            train = float(self.train_ratio.text()) / 100
            val = float(self.val_ratio.text()) / 100
            test = float(self.test_ratio.text()) / 100

            # 检查比例之和是否为100%左右（允许微小误差）
            if not (0.99 <= train + val + test <= 1.01):
                QMessageBox.warning(self, "比例错误", "比例之和必须为100%")
                return None

            return train, val, test
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字")
            return None


def export_all_results(project, image_editor, progress_bar):
    """导出所有图片的标注结果，包括原始图片并区分已标注和未标注"""
    if (not project or
            not project.processed_images or
            not project.output_dir):
        return

    # 显示比例设置对话框
    dialog = DatasetSplitDialog()
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return

    ratios = dialog.get_ratios()
    if not ratios:
        return
    train_ratio, val_ratio, test_ratio = ratios

    # 创建目录结构
    labeled_dir = os.path.join(project.output_dir, "labeled")
    unlabeled_dir = os.path.join(project.output_dir, "unlabeled")

    train_dir = os.path.join(labeled_dir, "train")
    val_dir = os.path.join(labeled_dir, "val")
    test_dir = os.path.join(labeled_dir, "test")

    for dir_path in [train_dir, val_dir, test_dir, unlabeled_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 分离已标注和未标注图片
    labeled_images = []
    unlabeled_images = []

    for image_path in project.image_paths:
        if image_path in project.process_status and project.process_status[image_path] in ["processed", "reviewed"]:
            labeled_images.append(image_path)
        else:
            unlabeled_images.append(image_path)

    # 随机打乱已标注图片顺序
    random.shuffle(labeled_images)
    total_labeled = len(labeled_images)

    # 计算各数据集数量
    train_count = int(total_labeled * train_ratio)
    val_count = int(total_labeled * val_ratio)
    # 剩下的作为测试集

    # 分配图片到各个数据集
    train_images = labeled_images[:train_count]
    val_images = labeled_images[train_count:train_count + val_count]
    test_images = labeled_images[train_count + val_count:]

    # 导出已标注图片及标注
    all_labeled = [(train_images, train_dir), (val_images, val_dir), (test_images, test_dir)]
    total = len(labeled_images) + len(unlabeled_images)
    current = 0

    for images, dest_dir in all_labeled:
        for image_path in images:
            # 导出标注结果
            export_single_result(project, image_path, image_editor, dest_dir)

            # 复制原始图片
            img_filename = os.path.basename(image_path)
            shutil.copy2(image_path, os.path.join(dest_dir, img_filename))

            current += 1
            progress_bar.setValue(int(current / total * 100))

    # 导出未标注图片
    for image_path in unlabeled_images:
        img_filename = os.path.basename(image_path)
        shutil.copy2(image_path, os.path.join(unlabeled_dir, img_filename))

        current += 1
        progress_bar.setValue(int(current / total * 100))

    QMessageBox.information(None, "完成",
                            f"所有图片已导出到 {project.output_dir}\n"
                            f"已标注: {len(labeled_images)} 张 (训练集: {len(train_images)}, 验证集: {len(val_images)}, 测试集: {len(test_images)})\n"
                            f"未标注: {len(unlabeled_images)} 张")
    progress_bar.setValue(0)


def export_single_result(project, image_path, image_editor, dest_dir=None):
    """导出单张图片的标注结果，包含彩色标注框"""
    if (not project or
            image_path not in project.processed_images):
        return

    # 如果未指定目标目录，使用项目的输出目录
    if not dest_dir:
        if not project.output_dir:
            return
        dest_dir = project.output_dir

    image, annotations = project.processed_images[image_path]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 保存标注后的图片，使用标签对应的颜色
    img_with_anns = image.copy()
    height, width = img_with_anns.shape[:2]

    for annot in annotations:
        x1, y1, x2, y2 = map(int, annot["box"])
        class_name = annot["class"]

        # 获取该标签的颜色
        color = image_editor.get_class_color(class_name)

        # 绘制边界框
        cv2.rectangle(img_with_anns, (x1, y1), (x2, y2), color, 2)

        # 绘制类别标签，文字颜色根据背景色自动调整
        r, g, b = color
        text_color = (0, 0, 0) if (r * 0.299 + g * 0.587 + b * 0.114) > 127 else (255, 255, 255)
        cv2.putText(img_with_anns, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img_with_anns, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 1)

    output_image_path = os.path.join(dest_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(output_image_path, img_with_anns)

    # 保存YOLO格式的标注文件
    output_txt_path = os.path.join(dest_dir, f"{base_name}.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for annot in annotations:
            x1, y1, x2, y2 = annot["box"]

            # 转换为YOLO格式：中心点坐标和宽高（归一化）
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            # 获取类别ID，使用项目中的标签ID
            class_id = 0
            if annot["class"] in project.class_names:
                class_id = project.class_names.index(annot["class"])

            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

