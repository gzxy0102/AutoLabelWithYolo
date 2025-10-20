import cv2
from PySide6.QtCore import (Qt, QRect, QPoint, Signal, QTimer)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QFont)
from PySide6.QtWidgets import (QLabel, QMessageBox, QMenu, QInputDialog)

from utils import generate_distinct_colors


class ImageEditor(QLabel):
    """图片编辑器，用于显示图片和编辑标注框，支持彩色标签"""
    annotation_updated = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 在__init__中定义所有实例属性
        self.image = None
        self.q_image = None
        self.annotations = []
        self.class_names = []
        self.class_colors = []
        self.current_box_idx = -1
        self.dragging = False
        self.drag_handle = None  # None, 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'center'
        self.last_pos = QPoint()
        self.box_offset = QPoint()
        
        # 添加一个标志，表示是否在拖拽过程中
        self.during_drag_operation = False

        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(640, 480)
        # 修正Qt.AlignCenter的引用
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    @property
    def current_annotation(self):
        """获取当前选中的标注框"""
        if 0 <= self.current_box_idx < len(self.annotations):
            return self.annotations[self.current_box_idx]
        return None

    @property
    def has_selection(self):
        """检查是否有选中的标注框"""
        return 0 <= self.current_box_idx < len(self.annotations)

    @property
    def has_annotations(self):
        """检查是否有标注"""
        return len(self.annotations) > 0

    def set_image(self, image):
        """设置显示的图片"""
        self.image = image
        self.update_q_image()
        self.annotations = []
        self.current_box_idx = -1
        self.update()

    def update_q_image(self):
        """将OpenCV图像转换为Qt图像"""
        if self.image is None:
            self.q_image = None
            return

        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # 修正Qt.KeepAspectRatio和Qt.SmoothTransformation的引用
        self.setPixmap(QPixmap.fromImage(self.q_image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def set_annotations(self, annotations):
        """设置标注信息"""
        self.annotations = annotations.copy()
        self.current_box_idx = -1
        self.update()

    def set_class_info(self, class_names, class_colors):
        """设置标签列表和对应的颜色"""
        self.class_names = class_names.copy()
        # 确保颜色数量与标签数量一致
        if len(class_colors) != len(class_names):
            self.class_colors = generate_distinct_colors(len(class_names))
        else:
            self.class_colors = class_colors.copy()

    @property
    def class_color_cache(self):
        """获取类颜色缓存"""
        return dict(zip(self.class_names, self.class_colors))

    def get_class_color(self, class_name):
        """获取标签对应的颜色，如果没有则返回默认颜色"""
        # 首先尝试从缓存中获取
        if class_name in self.class_color_cache:
            return self.class_color_cache[class_name]
        
        try:
            idx = self.class_names.index(class_name)
            color = self.class_colors[idx]
            return color
        except ValueError:
            return 0, 255, 0  # 默认绿色

    def paintEvent(self, event):
        """绘制事件，用于显示图片和带颜色的标注框"""
        super().paintEvent(event)
        if self.q_image is None or not self.annotations:
            return

        painter = QPainter(self)
        # 修正QPainter.Antialiasing的引用
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 计算缩放比例
        pixmap = self.pixmap()
        if pixmap.isNull():
            return

        # 计算图片在 QLabel 中的偏移量（居中显示时的偏移）
        pixmap_x = (self.width() - pixmap.width()) // 2
        pixmap_y = (self.height() - pixmap.height()) // 2

        scale_x = pixmap.width() / self.image.shape[1]
        scale_y = pixmap.height() / self.image.shape[0]

        # 绘制所有标注框，使用各自标签的颜色
        # 使用缓存的class_colors字典以提高查找性能
        for i, annot in enumerate(self.annotations):
            x1, y1, x2, y2 = annot["box"]
            x1_scaled = x1 * scale_x + pixmap_x
            y1_scaled = y1 * scale_y + pixmap_y
            x2_scaled = x2 * scale_x + pixmap_x
            y2_scaled = y2 * scale_y + pixmap_y

            # 获取该标签的颜色
            class_name = annot["class"]
            if class_name in self.class_color_cache:
                r, g, b = self.class_color_cache[class_name]
            else:
                r, g, b = self.get_class_color(class_name)
            
            # 选中的框使用稍微亮一点的颜色
            if i == self.current_box_idx:
                r = min(255, int(r * 1.2))
                g = min(255, int(g * 1.2))
                b = min(255, int(b * 1.2))

            color = QColor(r, g, b)
            pen = QPen(color, 2)
            painter.setPen(pen)
            rect = QRect(int(x1_scaled), int(y1_scaled),
                         int(x2_scaled - x1_scaled), int(y2_scaled - y1_scaled))
            painter.drawRect(rect)

            # 绘制类别标签，背景使用标签颜色
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            text_width = len(annot["class"]) * 12
            text_rect = QRect(int(x1_scaled), int(y1_scaled) - 20, text_width, 20)
            painter.fillRect(text_rect, color)
            # 根据背景亮度自动选择文字颜色
            luminance = (r * 0.299 + g * 0.587 + b * 0.114)
            text_color = QColor(0, 0, 0) if luminance > 127 else QColor(255, 255, 255)
            painter.setPen(text_color)
            painter.drawText(int(x1_scaled), int(y1_scaled) - 5,
                             f"{annot['class']} ({annot['confidence']:.2f})")

        # 如果有选中的框，绘制控制点
        if 0 <= self.current_box_idx < len(self.annotations):
            annot = self.annotations[self.current_box_idx]
            x1, y1, x2, y2 = annot["box"]
            x1_scaled = x1 * scale_x + pixmap_x
            y1_scaled = y1 * scale_y + pixmap_y
            x2_scaled = x2 * scale_x + pixmap_x
            y2_scaled = y2 * scale_y + pixmap_y

            # 绘制四个角的控制点，使用黄色
            control_size = 8
            points = [
                (x1_scaled, y1_scaled),  # top-left
                (x2_scaled, y1_scaled),  # top-right
                (x1_scaled, y2_scaled),  # bottom-left
                (x2_scaled, y2_scaled)  # bottom-right
            ]

            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(QColor(255, 255, 255))
            for (x, y) in points:
                painter.drawEllipse(int(x - control_size / 2), int(y - control_size / 2),
                                    control_size, control_size)

    def mousePressEvent(self, event):
        """鼠标按下事件，用于选择和拖动标注框"""
        if self.q_image is None or not self.annotations:
            return super().mousePressEvent(event)

        pos = event.pos()
        pixmap = self.pixmap()
        if pixmap.isNull():
            return None

        # 计算图片在 QLabel 中的偏移量（居中显示时的偏移）
        pixmap_x = (self.width() - pixmap.width()) // 2
        pixmap_y = (self.height() - pixmap.height()) // 2

        # 计算缩放比例
        scale_x = pixmap.width() / self.image.shape[1]
        scale_y = pixmap.height() / self.image.shape[0]
        inv_scale_x = 1 / scale_x
        inv_scale_y = 1 / scale_y

        # 检查是否点击了某个标注框
        self.current_box_idx = -1
        self.drag_handle = None

        for i, annot in enumerate(self.annotations):
            x1, y1, x2, y2 = annot["box"]
            x1_scaled = x1 * scale_x + pixmap_x
            y1_scaled = y1 * scale_y + pixmap_y
            x2_scaled = x2 * scale_x + pixmap_x
            y2_scaled = y2 * scale_y + pixmap_y

            # 检查是否点击了控制点
            control_size = 10
            handles = [
                (x1_scaled, y1_scaled, 'top_left'),
                (x2_scaled, y1_scaled, 'top_right'),
                (x1_scaled, y2_scaled, 'bottom_left'),
                (x2_scaled, y2_scaled, 'bottom_right')
            ]

            for (hx, hy, handle) in handles:
                if (abs(pos.x() - hx) <= control_size and
                        abs(pos.y() - hy) <= control_size):
                    self.current_box_idx = i
                    self.drag_handle = handle
                    self.last_pos = pos
                    self.dragging = True
                    self.update()
                    return None

            # 检查是否点击了框内部
            if (x1_scaled <= pos.x() <= x2_scaled and
                    y1_scaled <= pos.y() <= y2_scaled):
                self.current_box_idx = i
                self.drag_handle = 'center'
                self.last_pos = pos
                self.box_offset = QPoint(
                    int(pos.x() - x1_scaled),
                    int(pos.y() - y1_scaled)
                )
                self.dragging = True
                self.update()
                return None

        super().mousePressEvent(event)
        return None

    def mouseMoveEvent(self, event):
        """鼠标移动事件，用于拖动标注框"""
        if not self.dragging or self.current_box_idx < 0 or self.current_box_idx >= len(self.annotations):
            return super().mouseMoveEvent(event)

        pos = event.pos()
        pixmap = self.pixmap()
        if pixmap.isNull():
            return None

        # 计算图片在 QLabel 中的偏移量（居中显示时的偏移）
        pixmap_x = (self.width() - pixmap.width()) // 2
        pixmap_y = (self.height() - pixmap.height()) // 2

        # 计算缩放比例
        scale_x = pixmap.width() / self.image.shape[1]
        scale_y = pixmap.height() / self.image.shape[0]
        inv_scale_x = 1 / scale_x
        inv_scale_y = 1 / scale_y

        annot = self.annotations[self.current_box_idx]
        x1, y1, x2, y2 = annot["box"]
        new_x1, new_y1, new_x2, new_y2 = (x1, y1, x2, y2)  # 默认保持原值
        if self.drag_handle == 'center':
            # 整体移动
            dx = (pos.x() - self.last_pos.x()) * inv_scale_x
            dy = (pos.y() - self.last_pos.y()) * inv_scale_y

            new_x1 = x1 + dx
            new_y1 = y1 + dy
            new_x2 = x2 + dx
            new_y2 = y2 + dy

        elif self.drag_handle == 'top_left':
            # 左上角
            new_x1 = (pos.x() - pixmap_x) * inv_scale_x
            new_y1 = (pos.y() - pixmap_y) * inv_scale_y
            new_x2 = x2
            new_y2 = y2

        elif self.drag_handle == 'top_right':
            # 右上角
            new_x1 = x1
            new_y1 = (pos.y() - pixmap_y) * inv_scale_y
            new_x2 = (pos.x() - pixmap_x) * inv_scale_x
            new_y2 = y2

        elif self.drag_handle == 'bottom_left':
            # 左下角
            new_x1 = (pos.x() - pixmap_x) * inv_scale_x
            new_y1 = y1
            new_x2 = x2
            new_y2 = (pos.y() - pixmap_y) * inv_scale_y

        elif self.drag_handle == 'bottom_right':
            # 右下角
            new_x1 = x1
            new_y1 = y1
            new_x2 = (pos.x() - pixmap_x) * inv_scale_x
            new_y2 = (pos.y() - pixmap_y) * inv_scale_y

        # 确保坐标有效且在图像范围内
        img_width = self.image.shape[1]
        img_height = self.image.shape[0]
        
        # 保证坐标在有效范围内
        new_x1 = max(0, min(new_x1, img_width))
        new_y1 = max(0, min(new_y1, img_height))
        new_x2 = max(0, min(new_x2, img_width))
        new_y2 = max(0, min(new_y2, img_height))
        
        # 保证左上角坐标小于右下角坐标
        if new_x1 > new_x2:
            new_x1, new_x2 = new_x2, new_x1
        if new_y1 > new_y2:
            new_y1, new_y2 = new_y2, new_y1
            
        # 避免框的宽高为0
        if new_x1 == new_x2:
            if new_x1 > 0:
                new_x1 -= 1
            else:
                new_x2 += 1
        if new_y1 == new_y2:
            if new_y1 > 0:
                new_y1 -= 1
            else:
                new_y2 += 1

        # 更新标注框
        self.annotations[self.current_box_idx]["box"] = (new_x1, new_y1, new_x2, new_y2)
        self.last_pos = pos
        
        # 标记正在进行拖拽操作
        self.during_drag_operation = True
        
        # 只更新界面，不发送信号
        self.update()
        return None

    def perform_update(self):
        """执行实际的更新操作"""
        if self.pending_update:
            # 如果有待处理的更新，重新启动定时器
            self.update_timer.start(16)
            self.pending_update = False
        else:
            # 执行更新
            self.update()
            if self.pending_annotations:
                self.annotation_updated.emit(self.pending_annotations)

    def update_annotation_area(self, old_x1, old_y1, old_x2, old_y2, new_x1, new_y1, new_x2, new_y2, scale_x, scale_y, pixmap_x, pixmap_y):
        """只更新标注框区域以提高性能"""
        # 计算旧框和新框的屏幕坐标
        old_x1_scaled = old_x1 * scale_x + pixmap_x
        old_y1_scaled = old_y1 * scale_y + pixmap_y
        old_x2_scaled = old_x2 * scale_x + pixmap_x
        old_y2_scaled = old_y2 * scale_y + pixmap_y
        
        new_x1_scaled = new_x1 * scale_x + pixmap_x
        new_y1_scaled = new_y1 * scale_y + pixmap_y
        new_x2_scaled = new_x2 * scale_x + pixmap_x
        new_y2_scaled = new_y2 * scale_y + pixmap_y
        
        # 计算需要更新的区域（包括旧框和新框）
        update_x1 = int(min(old_x1_scaled, new_x1_scaled) - 10)
        update_y1 = int(min(old_y1_scaled, new_y1_scaled) - 10)
        update_x2 = int(max(old_x2_scaled, new_x2_scaled) + 10)
        update_y2 = int(max(old_y2_scaled, new_y2_scaled) + 10)
        
        # 更新指定区域
        self.update(update_x1, update_y1, update_x2 - update_x1, update_y2 - update_y1)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.dragging = False
        
        # 拖拽结束，发送更新信号并重置标志
        if self.during_drag_operation:
            self.during_drag_operation = False
            self.annotation_updated.emit(self.annotations)
            
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        if self.q_image:
            # 修正Qt.KeepAspectRatio和Qt.SmoothTransformation的引用
            self.setPixmap(QPixmap.fromImage(self.q_image).scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        super().resizeEvent(event)
        # 触发重绘以确保标注框正确显示
        self.update()

    def contextMenuEvent(self, event):
        """右键菜单事件，用于删除标注框或修改标签"""
        if self.current_box_idx < 0 or self.current_box_idx >= len(self.annotations):
            return super().contextMenuEvent(event)

        # 创建上下文菜单
        menu = QMenu(self)

        # 修改标签动作
        modify_action = menu.addAction("修改标签")
        # 删除标注动作
        delete_action = menu.addAction("删除此标注")

        # 执行选中的动作
        action = menu.exec(self.mapToGlobal(event.pos()))
        if action == modify_action:
            self.modify_annotation_label()
        elif action == delete_action:
            self.delete_annotation()
        return None

    def modify_annotation_label(self):
        """修改标注标签"""
        if not self.class_names:
            QMessageBox.warning(self, "警告", "没有可用的标签，请先在项目设置中添加标签")
            return

        current_label = self.annotations[self.current_box_idx]["class"]
        # 从项目标签列表中选择
        label, ok = QInputDialog.getItem(
            self, "修改标签", "请选择标签:",
            self.class_names, self.class_names.index(current_label) if current_label in self.class_names else 0,
            False)

        if ok and label:
            self.annotations[self.current_box_idx]["class"] = label
            # 更新类别ID
            if label in self.class_names:
                self.annotations[self.current_box_idx]["class_id"] = self.class_names.index(label)
            self.update()
            self.annotation_updated.emit(self.annotations)

    def delete_annotation(self):
        """删除标注"""
        del self.annotations[self.current_box_idx]
        self.current_box_idx = -1
        self.update()
        self.annotation_updated.emit(self.annotations)
