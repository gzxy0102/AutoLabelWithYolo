import sys
import os
import cv2
from PySide6.QtWidgets import (QMainWindow, QPushButton, QLabel, QFileDialog,
                               QVBoxLayout, QHBoxLayout, QWidget, QCheckBox,
                               QProgressBar, QListWidget, QSplitter, QMessageBox,
                               QInputDialog, QListWidgetItem, QGroupBox, QFormLayout,
                               QComboBox, QLineEdit, QMenuBar, QMenu, QDialog,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QToolBar, QColorDialog)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QFont,
                           QIcon, QAction, QKeyEvent)
from PySide6.QtCore import (Qt, QRect, QPoint, Signal, QThread, QTimer,
                            Slot)

from project import Project
from annotation import AnnotationThread
from utils import generate_distinct_colors


class ClassManagementDialog(QDialog):
    """标签管理对话框，支持颜色设置"""

    def __init__(self, class_names, class_colors, parent=None):
        super().__init__(parent)

        # 在__init__中定义所有实例属性
        self.class_names = []
        self.class_colors = []
        self.class_table = None
        self.add_btn = None
        self.edit_btn = None
        self.color_btn = None
        self.remove_btn = None
        self.auto_color_btn = None
        self.ok_btn = None
        self.cancel_btn = None

        self.init_ui(class_names, class_colors)

    def init_ui(self, class_names, class_colors):
        self.setWindowTitle("标签管理")
        self.setGeometry(200, 200, 500, 300)

        self.class_names = class_names.copy()
        # 确保颜色数量与标签数量一致
        if len(class_colors) != len(class_names):
            self.class_colors = generate_distinct_colors(len(class_names))
        else:
            self.class_colors = [tuple(color) for color in class_colors]

        layout = QVBoxLayout()

        # 标签列表
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.setHorizontalHeaderLabels(["ID", "标签名称", "颜色"])
        # 修正QHeaderView.Stretch的引用
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # 修正QAbstractItemView.NoEditTriggers的引用
        self.class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # 设置列宽比例
        self.class_table.setColumnWidth(0, 50)
        self.class_table.setColumnWidth(2, 100)
        layout.addWidget(self.class_table)

        # 按钮布局
        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("添加标签")
        self.add_btn.clicked.connect(self.add_class)
        btn_layout.addWidget(self.add_btn)

        self.edit_btn = QPushButton("编辑标签")
        self.edit_btn.clicked.connect(self.edit_class)
        btn_layout.addWidget(self.edit_btn)

        self.color_btn = QPushButton("修改颜色")
        self.color_btn.clicked.connect(self.change_color)
        btn_layout.addWidget(self.color_btn)

        self.remove_btn = QPushButton("删除标签")
        self.remove_btn.clicked.connect(self.remove_class)
        btn_layout.addWidget(self.remove_btn)

        self.auto_color_btn = QPushButton("自动生成颜色")
        self.auto_color_btn.clicked.connect(self.auto_generate_colors)
        btn_layout.addWidget(self.auto_color_btn)

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.update_table()

    def update_table(self):
        """更新标签表格，包括颜色显示"""
        self.class_table.setRowCount(len(self.class_names))
        for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
            # ID列
            self.class_table.setItem(i, 0, QTableWidgetItem(str(i)))

            # 标签名称列
            self.class_table.setItem(i, 1, QTableWidgetItem(class_name))

            # 颜色列 - 显示颜色方块
            color_widget = QWidget()
            color_layout = QHBoxLayout(color_widget)
            color_layout.setContentsMargins(5, 5, 5, 5)

            color_label = QLabel()
            color_label.setFixedSize(30, 30)
            color_label.setStyleSheet(
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid #ccc;")

            color_layout.addWidget(color_label)
            color_layout.addStretch()
            self.class_table.setCellWidget(i, 2, color_widget)

    def add_class(self):
        """添加新标签，自动生成颜色"""
        class_name, ok = QInputDialog.getText(self, "添加标签", "请输入标签名称:")
        if ok and class_name and class_name not in self.class_names:
            self.class_names.append(class_name)
            # 生成新的颜色
            new_colors = generate_distinct_colors(len(self.class_names))
            self.class_colors = new_colors
            self.update_table()

    def edit_class(self):
        """编辑选中的标签"""
        current_row = self.class_table.currentRow()
        if current_row >= 0 and current_row < len(self.class_names):
            old_name = self.class_names[current_row]
            new_name, ok = QInputDialog.getText(
                self, "编辑标签", "请输入新的标签名称:", text=old_name)
            if ok and new_name and new_name not in self.class_names:
                self.class_names[current_row] = new_name
                self.update_table()

    def change_color(self):
        """修改选中标签的颜色"""
        current_row = self.class_table.currentRow()
        if current_row >= 0 and current_row < len(self.class_names):
            current_color = self.class_colors[current_row]
            # 打开颜色选择对话框
            color = QColorDialog.getColor(
                QColor(current_color[0], current_color[1], current_color[2]),
                self, "选择标签颜色"
            )
            if color.isValid():
                self.class_colors[current_row] = (color.red(), color.green(), color.blue())
                self.update_table()

    def remove_class(self):
        """删除选中的标签"""
        current_row = self.class_table.currentRow()
        if current_row >= 0 and current_row < len(self.class_names):
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除标签 '{self.class_names[current_row]}' 吗?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                del self.class_names[current_row]
                del self.class_colors[current_row]
                # 重新生成所有颜色，确保视觉差异
                self.class_colors = generate_distinct_colors(len(self.class_names))
                self.update_table()

    def auto_generate_colors(self):
        """为所有标签自动生成颜色"""
        self.class_colors = generate_distinct_colors(len(self.class_names))
        self.update_table()

    def get_class_info(self):
        """返回修改后的标签列表和颜色列表"""
        return self.class_names, self.class_colors


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

        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(640, 480)
        # 修正Qt.AlignCenter的引用
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

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

    def get_class_color(self, class_name):
        """获取标签对应的颜色，如果没有则返回默认颜色"""
        try:
            idx = self.class_names.index(class_name)
            return self.class_colors[idx]
        except ValueError:
            return (0, 255, 0)  # 默认绿色

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

        scale_x = pixmap.width() / self.image.shape[1]
        scale_y = pixmap.height() / self.image.shape[0]

        # 绘制所有标注框，使用各自标签的颜色
        for i, annot in enumerate(self.annotations):
            x1, y1, x2, y2 = annot["box"]
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # 获取该标签的颜色
            r, g, b = self.get_class_color(annot["class"])
            # 选中的框使用稍微亮一点的颜色
            if i == self.current_box_idx:
                r = min(255, int(r * 1.2))
                g = min(255, int(g * 1.2))
                b = min(255, int(b * 1.2))

            color = QColor(r, g, b)
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(QRect(int(x1_scaled), int(y1_scaled),
                                   int(x2_scaled - x1_scaled), int(y2_scaled - y1_scaled)))

            # 绘制类别标签，背景使用标签颜色
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            painter.fillRect(int(x1_scaled), int(y1_scaled) - 20,
                             len(annot["class"]) * 12, 20, color)
            # 根据背景亮度自动选择文字颜色
            luminance = (r * 0.299 + g * 0.587 + b * 0.114)
            text_color = QColor(0, 0, 0) if luminance > 127 else QColor(255, 255, 255)
            painter.setPen(text_color)
            painter.drawText(int(x1_scaled), int(y1_scaled) - 5,
                             f"{annot['class']} ({annot['confidence']:.2f})")

        # 如果有选中的框，绘制控制点
        if self.current_box_idx >= 0 and self.current_box_idx < len(self.annotations):
            annot = self.annotations[self.current_box_idx]
            x1, y1, x2, y2 = annot["box"]
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # 绘制四个角的控制点，使用黄色
            control_size = 8
            points = [
                (x1_scaled, y1_scaled),  # top-left
                (x2_scaled, y1_scaled),  # top-right
                (x1_scaled, y2_scaled),  # bottom-left
                (x2_scaled, y2_scaled)  # bottom-right
            ]

            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(QColor(255, 255, 0))
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
            return

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
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

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

        # 计算缩放比例
        scale_x = pixmap.width() / self.image.shape[1]
        scale_y = pixmap.height() / self.image.shape[0]
        inv_scale_x = 1 / scale_x
        inv_scale_y = 1 / scale_y

        annot = self.annotations[self.current_box_idx]
        x1, y1, x2, y2 = annot["box"]
        new_x1, new_y1, new_x2, new_y2 = (None, None, None, None)
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
            new_x1 = pos.x() * inv_scale_x
            new_y1 = pos.y() * inv_scale_y
            new_x2 = x2
            new_y2 = y2

        elif self.drag_handle == 'top_right':
            # 右上角
            new_x1 = x1
            new_y1 = pos.y() * inv_scale_y
            new_x2 = pos.x() * inv_scale_x
            new_y2 = y2

        elif self.drag_handle == 'bottom_left':
            # 左下角
            new_x1 = pos.x() * inv_scale_x
            new_y1 = y1
            new_x2 = x2
            new_y2 = pos.y() * inv_scale_y

        elif self.drag_handle == 'bottom_right':
            # 右下角
            new_x1 = x1
            new_y1 = y1
            new_x2 = pos.x() * inv_scale_x
            new_y2 = pos.y() * inv_scale_y

        # 确保坐标有效
        new_x1 = max(0, min(new_x1, self.image.shape[1]))
        new_y1 = max(0, min(new_y1, self.image.shape[0]))
        new_x2 = max(new_x1, min(new_x2, self.image.shape[1]))
        new_y2 = max(new_y1, min(new_y2, self.image.shape[0]))

        # 更新标注框
        self.annotations[self.current_box_idx]["box"] = (new_x1, new_y1, new_x2, new_y2)
        self.last_pos = pos
        self.update()
        self.annotation_updated.emit(self.annotations)
        return None

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.dragging = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        if self.q_image:
            # 修正Qt.KeepAspectRatio和Qt.SmoothTransformation的引用
            self.setPixmap(QPixmap.fromImage(self.q_image).scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        super().resizeEvent(event)

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


class YOLOAnnotationTool(QMainWindow):
    """带项目管理、彩色标签的YOLO自动标注工具主窗口"""

    def __init__(self):
        super().__init__()

        # 在__init__中定义所有实例属性
        self.current_project = None
        self.current_image_idx = -1
        self.annotation_thread = None

        # UI组件
        self.project_info_group = None
        self.project_info_layout = None
        self.project_name_label = None
        self.project_path_label = None
        self.class_count_label = None

        self.config_group = None
        self.config_layout = None
        self.image_dir_label = None
        self.select_image_dir_btn = None
        self.model_path_label = None
        self.select_model_btn = None
        self.output_dir_label = None
        self.select_output_dir_btn = None
        self.review_checkbox = None
        self.manage_classes_btn = None
        self.process_btn = None
        self.progress_bar = None
        self.complete_btn = None

        self.splitter = None
        self.image_list = None
        self.image_editor = None

        self.btn_layout = None
        self.prev_btn = None
        self.next_btn = None
        self.save_btn = None
        self.export_all_btn = None

        self.init_ui()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("YOLO自动标注工具 - 无项目")
        self.setGeometry(100, 100, 1200, 800)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建工具栏
        self.create_tool_bar()

        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 创建项目信息区域
        self.project_info_group = QGroupBox("项目信息")
        self.project_info_layout = QFormLayout()

        self.project_name_label = QLabel("无项目")
        self.project_path_label = QLabel("未保存")
        self.class_count_label = QLabel("0个标签")

        self.project_info_layout.addRow("项目名称:", self.project_name_label)
        self.project_info_layout.addRow("项目路径:", self.project_path_label)
        self.project_info_layout.addRow("标签数量:", self.class_count_label)

        self.project_info_group.setLayout(self.project_info_layout)
        main_layout.addWidget(self.project_info_group)

        # 创建配置区域
        self.config_group = QGroupBox("配置选项")
        self.config_layout = QFormLayout()

        # 图片目录选择
        self.image_dir_label = QLabel("未选择")
        self.select_image_dir_btn = QPushButton("选择图片目录")
        self.select_image_dir_btn.clicked.connect(self.select_image_dir)
        self.config_layout.addRow("图片目录:", self.image_dir_label)
        self.config_layout.addRow(self.select_image_dir_btn)

        # 模型文件选择
        self.model_path_label = QLabel("未选择")
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.clicked.connect(self.select_model_file)
        self.config_layout.addRow("模型文件:", self.model_path_label)
        self.config_layout.addRow(self.select_model_btn)

        # 输出目录选择
        self.output_dir_label = QLabel("未选择")
        self.select_output_dir_btn = QPushButton("选择输出目录")
        self.select_output_dir_btn.clicked.connect(self.select_output_dir)
        self.config_layout.addRow("输出目录:", self.output_dir_label)
        self.config_layout.addRow(self.select_output_dir_btn)

        # 人工复判选项
        self.review_checkbox = QCheckBox("需要人工复判")
        self.review_checkbox.setChecked(True)
        self.review_checkbox.clicked.connect(self.on_review_toggled)
        self.config_layout.addRow(self.review_checkbox)

        # 标签管理按钮
        self.manage_classes_btn = QPushButton("管理标签")
        self.manage_classes_btn.clicked.connect(self.manage_classes)
        self.config_layout.addRow(self.manage_classes_btn)

        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        self.config_layout.addRow(self.process_btn)

        # 完成标注按钮
        self.complete_btn = QPushButton("完成标注 (Enter)")
        self.complete_btn.clicked.connect(self.complete_annotation)
        self.complete_btn.setEnabled(False)
        self.config_layout.addRow(self.complete_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.config_layout.addRow("进度:", self.progress_bar)

        self.config_group.setLayout(self.config_layout)
        main_layout.addWidget(self.config_group)

        # 创建工作区域
        # 修正Qt.Horizontal的引用
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧图片列表
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.on_image_double_clicked)
        self.splitter.addWidget(self.image_list)

        # 右侧图片编辑区
        self.image_editor = ImageEditor()
        self.image_editor.annotation_updated.connect(self.on_annotation_updated)
        self.splitter.addWidget(self.image_editor)

        # 底部按钮区域
        self.btn_layout = QHBoxLayout()

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)

        self.save_btn = QPushButton("保存当前标注")
        self.save_btn.clicked.connect(self.save_current_annotation)
        self.save_btn.setEnabled(False)

        self.export_all_btn = QPushButton("导出所有结果")
        self.export_all_btn.clicked.connect(self.export_all_results)
        self.export_all_btn.setEnabled(False)

        self.btn_layout.addWidget(self.prev_btn)
        self.btn_layout.addWidget(self.next_btn)
        self.btn_layout.addWidget(self.save_btn)
        self.btn_layout.addWidget(self.export_all_btn)

        # 添加到主布局
        main_layout.addWidget(self.splitter, 1)
        main_layout.addLayout(self.btn_layout)

        self.setCentralWidget(main_widget)

        # 初始禁用所有按钮，直到创建或打开项目
        self.set_widgets_enabled(False)

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # 修正Qt.Key_Return和Qt.Key_Enter的引用
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.complete_btn.isEnabled():
                self.complete_annotation()
        # 修正Qt.Key_Delete的引用
        elif event.key() == Qt.Key.Key_Delete:
            if self.image_editor.current_box_idx >= 0:
                self.image_editor.delete_annotation()
        super().keyPressEvent(event)

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 项目菜单
        project_menu = menubar.addMenu("项目")

        new_project_action = QAction("新建项目", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)
        project_menu.addAction(new_project_action)

        open_project_action = QAction("打开项目", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.open_project)
        project_menu.addAction(open_project_action)

        save_project_action = QAction("保存项目", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self.save_project)
        project_menu.addAction(save_project_action)

        save_as_project_action = QAction("另存为", self)
        save_as_project_action.triggered.connect(self.save_project_as)
        project_menu.addAction(save_as_project_action)

        project_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        project_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")

        manage_classes_action = QAction("管理标签", self)
        manage_classes_action.triggered.connect(self.manage_classes)
        edit_menu.addAction(manage_classes_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("工具栏", self)
        self.addToolBar(toolbar)

        new_project_action = QAction("新建项目", self)
        new_project_action.triggered.connect(self.new_project)
        toolbar.addAction(new_project_action)

        open_project_action = QAction("打开项目", self)
        open_project_action.triggered.connect(self.open_project)
        toolbar.addAction(open_project_action)

        save_project_action = QAction("保存项目", self)
        save_project_action.triggered.connect(self.save_project)
        toolbar.addAction(save_project_action)

        toolbar.addSeparator()

        process_action = QAction("开始处理", self)
        process_action.triggered.connect(self.start_processing)
        toolbar.addAction(process_action)

        export_action = QAction("导出结果", self)
        export_action.triggered.connect(self.export_all_results)
        toolbar.addAction(export_action)

    def set_widgets_enabled(self, enabled):
        """设置控件是否可用"""
        self.select_image_dir_btn.setEnabled(enabled)
        self.select_model_btn.setEnabled(enabled)
        self.select_output_dir_btn.setEnabled(enabled)
        self.review_checkbox.setEnabled(enabled)
        self.manage_classes_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled and self.check_process_ready())
        self.export_all_btn.setEnabled((enabled and self.current_project.output_dir != "" and
                                        len(self.current_project.processed_images.keys()) > 0) if self.current_project is not None else False)
        self.prev_btn.setEnabled(enabled and self.current_image_idx > 0)
        self.next_btn.setEnabled(enabled and self.current_project and
                                 self.current_image_idx < len(self.current_project.image_paths) - 1)
        self.save_btn.setEnabled((enabled and self.current_image_idx >= 0 and
                                  self.current_project.output_dir != "") if self.current_project is not None else False)
        self.complete_btn.setEnabled((enabled and self.current_image_idx >= 0
                                      and self.current_project.review_required) if self.current_project is not None else False)

    def new_project(self):
        """创建新项目"""
        # 询问是否保存当前项目
        if self.current_project and self.current_project.path:
            reply = QMessageBox.question(
                self, "保存当前项目", "是否保存当前项目?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                self.save_project()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        # 获取项目名称
        project_name, ok = QInputDialog.getText(self, "新建项目", "请输入项目名称:")
        if not ok or not project_name:
            return

        # 获取标签信息（新增部分）
        default_classes = "person,car,bike,bus,truck"  # 默认标签示例
        class_input, ok = QInputDialog.getText(
            self, "设置标签",
            f"请输入标签名称，用逗号分隔:\n(示例: {default_classes})",
            text=default_classes
        )
        if not ok:
            return  # 用户取消创建项目

        class_names = []
        if class_input:
            # 分割并处理每个标签（去空格、过滤空字符串）
            class_names = [cls.strip() for cls in class_input.split(',') if cls.strip()]
        else:
            class_input = default_classes.split(",")

        # 获取项目保存路径
        project_path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", f"{project_name}.yap", "YOLO标注项目 (*.yap);;所有文件 (*)")
        if not project_path:
            return

        # 创建新项目
        self.current_project = Project(project_name, project_path)
        # 设置初始标签和颜色（新增部分）
        self.current_project.class_names = class_names
        self.current_project.class_colors = generate_distinct_colors(len(class_names))

        self.current_image_idx = -1
        self.update_project_info()
        self.update_image_list()
        self.image_editor.set_image(None)
        self.image_editor.set_annotations([])
        self.image_editor.set_class_info(
            self.current_project.class_names,
            self.current_project.class_colors
        )
        self.set_widgets_enabled(True)

    def open_project(self):
        """打开现有项目"""
        # 询问是否保存当前项目
        if self.current_project and self.current_project.path:
            reply = QMessageBox.question(
                self, "保存当前项目", "是否保存当前项目?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                self.save_project()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        # 选择项目文件
        project_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "YOLO标注项目 (*.yap);;所有文件 (*)")
        if not project_path:
            return

        # 加载项目，恢复缓存信息
        self.current_project = Project()
        if self.current_project.load(project_path):
            # 恢复上次处理的位置
            self.current_image_idx = self.current_project.last_processed_index
            self.update_project_info()
            self.update_image_list()
            self.image_editor.set_image(None)
            self.image_editor.set_annotations([])
            self.image_editor.set_class_info(
                self.current_project.class_names,
                self.current_project.class_colors
            )
            self.review_checkbox.setChecked(self.current_project.review_required)
            self.set_widgets_enabled(True)

            # 更新配置显示
            self.image_dir_label.setText(self.current_project.image_dir)
            self.model_path_label.setText(self.current_project.model_path)
            self.output_dir_label.setText(self.current_project.output_dir)

            # 如果有上次处理的图片，显示它
            if 0 <= self.current_image_idx < len(self.current_project.image_paths):
                self.show_current_image()
                self.update_nav_buttons()
        else:
            QMessageBox.critical(self, "错误", "加载项目失败")
            self.current_project = None
            self.set_widgets_enabled(False)

    def save_project(self):
        """保存当前项目，包括缓存的标注信息"""
        if not self.current_project:
            return False

        # 保存当前处理位置
        if self.current_image_idx >= 0:
            self.current_project.last_processed_index = self.current_image_idx

        if not self.current_project.path:
            return self.save_project_as()

        return self.current_project.save()

    def save_project_as(self):
        """另存为新项目"""
        if not self.current_project:
            return False

        project_path, _ = QFileDialog.getSaveFileName(
            self, "另存项目", self.current_project.path or f"{self.current_project.name}.yap",
            "YOLO标注项目 (*.yap);;所有文件 (*)")
        if project_path:
            return self.current_project.save(project_path)
        return False

    def update_project_info(self):
        """更新项目信息显示"""
        if not self.current_project:
            self.setWindowTitle("YOLO自动标注工具 - 无项目")
            self.project_name_label.setText("无项目")
            self.project_path_label.setText("未保存")
            self.class_count_label.setText("0个标签")
            return

        self.setWindowTitle(f"YOLO自动标注工具 - {self.current_project.name}")
        self.project_name_label.setText(self.current_project.name)
        self.project_path_label.setText(self.current_project.path or "未保存")
        self.class_count_label.setText(f"{len(self.current_project.class_names)}个标签")

    def manage_classes(self):
        """管理标签，包括颜色设置"""
        if not self.current_project:
            QMessageBox.warning(self, "警告", "请先创建或打开一个项目")
            return

        # 打开标签管理对话框，传入当前标签和颜色
        dialog = ClassManagementDialog(
            self.current_project.class_names,
            self.current_project.class_colors,
            self
        )
        if dialog.exec():
            new_classes, new_colors = dialog.get_class_info()
            self.current_project.class_names = new_classes
            self.current_project.class_colors = new_colors
            self.image_editor.set_class_info(new_classes, new_colors)
            self.class_count_label.setText(f"{len(new_classes)}个标签")
            self.save_project()  # 自动保存项目

    def select_image_dir(self):
        """选择图片目录"""
        if not self.current_project:
            return

        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.current_project.image_dir = dir_path
            self.image_dir_label.setText(dir_path)

            # 获取目录中的图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            self.current_project.image_paths = []
            for file in os.listdir(dir_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    self.current_project.image_paths.append(os.path.join(dir_path, file))

            # 更新图片列表
            self.update_image_list()
            self.check_process_ready()
            self.save_project()  # 自动保存项目

    def update_image_list(self):
        """更新图片列表"""
        self.image_list.clear()
        if not self.current_project:
            return

        for path in self.current_project.image_paths:
            item = QListWidgetItem(os.path.basename(path))
            # 修正Qt.UserRole的引用
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.image_list.addItem(item)

            # 标记已处理和已审查的图片
            if path in self.current_project.process_status:
                status = self.current_project.process_status[path]
                if status == "processed":
                    item.setForeground(QColor(0, 128, 0))  # 已处理 - 绿色
                elif status == "reviewed":
                    item.setForeground(QColor(0, 0, 128))  # 已审查 - 蓝色

    def select_model_file(self):
        """选择模型文件"""
        if not self.current_project:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "YOLO模型文件 (*.pt);;所有文件 (*)")
        if file_path:
            self.current_project.model_path = file_path
            self.model_path_label.setText(file_path)
            self.check_process_ready()
            self.save_project()  # 自动保存项目

    def select_output_dir(self):
        """选择输出目录"""
        if not self.current_project:
            return

        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.current_project.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.export_all_btn.setEnabled(len(self.current_project.processed_images) > 0)
            self.save_btn.setEnabled(self.current_image_idx >= 0)
            self.save_project()  # 自动保存项目

    def on_review_toggled(self):
        """人工复判选项变化"""
        if self.current_project:
            self.current_project.review_required = self.review_checkbox.isChecked()
            self.complete_btn.setEnabled(self.current_image_idx >= 0 and
                                         self.current_project.review_required)
            self.save_project()  # 自动保存项目

    def check_process_ready(self):
        """检查是否可以开始处理"""
        if not self.current_project:
            return False

        ready = (bool(self.current_project.image_dir) and
                 bool(self.current_project.model_path) and
                 len(self.current_project.image_paths) > 0)
        self.process_btn.setEnabled(ready)
        return ready

    def start_processing(self):
        """开始处理图片"""
        if not self.current_project or not self.check_process_ready():
            return

        # 禁用按钮
        self.set_widgets_enabled(False)

        # 确定开始处理的位置（从上次处理的位置继续）
        start_idx = self.current_project.last_processed_index
        images_to_process = self.current_project.image_paths[start_idx:]

        if not images_to_process:
            QMessageBox.information(self, "提示", "所有图片已处理完毕")
            self.set_widgets_enabled(True)
            return

        # 启动标注线程
        self.annotation_thread = AnnotationThread(
            images_to_process,
            self.current_project.model_path,
            self.current_project.class_names)
        self.annotation_thread.progress_updated.connect(self.update_progress)
        self.annotation_thread.image_processed.connect(self.on_image_processed)
        self.annotation_thread.finished.connect(self.on_processing_finished)
        self.annotation_thread.start()

    def update_progress(self, value):
        """更新进度条"""
        # 计算全局进度
        start_idx = self.current_project.last_processed_index
        total = len(self.current_project.image_paths)
        processed = len(self.current_project.image_paths) - len(self.current_project.image_paths[start_idx:])
        current = int(processed / total * 100) + int(
            value * len(self.current_project.image_paths[start_idx:]) / total / 100)
        self.progress_bar.setValue(current)

    def on_image_processed(self, image_path, image, annotations):
        """处理完一张图片后的回调"""
        if not self.current_project:
            return

        self.current_project.processed_images[image_path] = (image, annotations)
        self.current_project.process_status[image_path] = "processed"

        # 更新图片列表状态
        self.update_image_list()

        # 获取当前图片索引
        if image_path in self.current_project.image_paths:
            self.current_image_idx = self.current_project.image_paths.index(image_path)
            self.current_project.last_processed_index = self.current_image_idx

            # 如果需要人工复判，显示当前图片并暂停处理
            if self.current_project.review_required:
                # 暂停线程
                if self.annotation_thread:
                    self.annotation_thread.pause()

                # 显示当前图片
                self.show_current_image()
                self.update_nav_buttons()
                self.set_widgets_enabled(True)
                self.complete_btn.setEnabled(True)

    def on_processing_finished(self):
        """所有图片处理完成后的回调"""
        # 启用按钮
        self.set_widgets_enabled(True)
        self.save_project()  # 保存处理结果

        # 如果不需要人工复判，自动导出结果
        if (not self.current_project.review_required and
                self.current_project.output_dir):
            self.export_all_results()
            QMessageBox.information(self, "完成", "所有图片处理完成并已导出结果")
        elif len(self.current_project.processed_images) > 0:
            QMessageBox.information(self, "完成", "所有图片处理完成")
            # 如果还没有显示任何图片，显示第一张
            if self.current_image_idx == -1:
                self.current_image_idx = 0
                self.show_current_image()
                self.update_nav_buttons()

    def show_current_image(self):
        """显示当前图片和带颜色的标注"""
        if (not self.current_project or
                self.current_image_idx < 0 or
                self.current_image_idx >= len(self.current_project.image_paths)):
            return

        image_path = self.current_project.image_paths[self.current_image_idx]
        if image_path in self.current_project.processed_images:
            image, annotations = self.current_project.processed_images[image_path]

            # 如果图像为None（从项目加载的情况），重新加载
            if image is None:
                image = cv2.imread(image_path)
                self.current_project.processed_images[image_path] = (image, annotations)

            self.image_editor.set_image(image)
            self.image_editor.set_annotations(annotations)
            self.save_btn.setEnabled(True)

            # 高亮显示列表中的当前项
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == image_path:
                    self.image_list.setCurrentItem(item)
                    break

    def on_image_double_clicked(self, item):
        """双击图片列表项"""
        if not self.current_project:
            return

        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path in self.current_project.image_paths:
            self.current_image_idx = self.current_project.image_paths.index(image_path)
            self.show_current_image()
            self.update_nav_buttons()

    def prev_image(self):
        """显示上一张图片"""
        if self.current_project and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.show_current_image()
            self.update_nav_buttons()

    def next_image(self):
        """显示下一张图片"""
        if (self.current_project and
                self.current_image_idx < len(self.current_project.image_paths) - 1):
            self.current_image_idx += 1

            # 如果下一张未处理且需要自动处理，启动处理
            image_path = self.current_project.image_paths[self.current_image_idx]
            if (self.current_project.review_required and
                    image_path not in self.current_project.process_status):
                self.start_processing()
            else:
                self.show_current_image()
                self.update_nav_buttons()

    def update_nav_buttons(self):
        """更新导航按钮状态"""
        if not self.current_project:
            return

        self.prev_btn.setEnabled(self.current_image_idx > 0)
        self.next_btn.setEnabled(self.current_image_idx < len(self.current_project.image_paths) - 1)
        self.complete_btn.setEnabled(self.current_project.review_required)

    def complete_annotation(self):
        """完成当前图片标注，进入下一张"""
        if (not self.current_project or
                self.current_image_idx < 0 or
                self.current_image_idx >= len(self.current_project.image_paths)):
            return

        # 标记为已审查
        image_path = self.current_project.image_paths[self.current_image_idx]
        self.current_project.process_status[image_path] = "reviewed"
        self.save_current_annotation()
        self.update_image_list()

        # 移动到下一张
        if self.current_image_idx < len(self.current_project.image_paths) - 1:
            self.current_image_idx += 1
            self.current_project.last_processed_index = self.current_image_idx

            # 检查下一张是否已处理
            next_image_path = self.current_project.image_paths[self.current_image_idx]
            if next_image_path in self.current_project.processed_images:
                # 已处理，直接显示
                self.show_current_image()
                self.update_nav_buttons()
            else:
                # 未处理，自动处理
                self.start_processing()
        else:
            # 已经是最后一张
            QMessageBox.information(self, "完成", "所有图片已处理完毕")
            self.current_image_idx = len(self.current_project.image_paths) - 1
            self.update_nav_buttons()

    def on_annotation_updated(self, annotations):
        """标注更新时的回调"""
        if (not self.current_project or
                self.current_image_idx < 0 or
                self.current_image_idx >= len(self.current_project.image_paths)):
            return

        image_path = self.current_project.image_paths[self.current_image_idx]
        if image_path in self.current_project.processed_images:
            image, _ = self.current_project.processed_images[image_path]
            self.current_project.processed_images[image_path] = (image, annotations)
            self.save_project()  # 自动保存修改

    def save_current_annotation(self):
        """保存当前图片的标注"""
        if (not self.current_project or
                self.current_image_idx < 0 or
                self.current_image_idx >= len(self.current_project.image_paths) or
                not self.current_project.output_dir):
            return

        image_path = self.current_project.image_paths[self.current_image_idx]
        if image_path in self.current_project.processed_images:
            from annotation import export_single_result
            export_single_result(self.current_project, image_path, self.image_editor)
            QMessageBox.information(self, "成功", f"已保存 {os.path.basename(image_path)} 的标注结果")

    def export_all_results(self):
        """导出所有结果"""
        if (not self.current_project or
                not self.current_project.processed_images or
                not self.current_project.output_dir):
            return

        from annotation import export_all_results
        export_all_results(self.current_project, self.image_editor, self.progress_bar)

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", "YOLO自动标注工具\n版本 1.2\n支持项目管理、自定义标签和彩色标注")

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 检查是否正在处理
        if self.annotation_thread and self.annotation_thread.isRunning():
            reply = QMessageBox.question(self, "确认",
                                         "正在处理图片，确定要退出吗？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.annotation_thread.stop()
                event.accept()
            else:
                event.ignore()
                return

        # 检查是否需要保存项目
        if self.current_project:
            reply = QMessageBox.question(self, "保存项目",
                                         "是否保存当前项目？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                self.save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
