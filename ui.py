import logging
import os
import random
import shutil
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
from PySide6.QtCore import (Qt, QTimer)
from PySide6.QtGui import (QColor, QAction, QIcon)
from PySide6.QtWidgets import (QMainWindow, QPushButton, QLabel, QFileDialog,
                               QVBoxLayout, QHBoxLayout, QWidget, QProgressBar, QListWidget, QSplitter, QMessageBox,
                               QInputDialog, QListWidgetItem, QGroupBox, QFormLayout,
                               QDialog,
                               QToolBar)
from ultralytics import YOLO

from dialogs import DatasetSplitDialog, ClassManagementDialog
from image_editor import ImageEditor
from project import Project
from utils import generate_distinct_colors

# 配置日志
logger = logging.getLogger(__name__)

# 应用程序版本信息
APP_VERSION = "1.0.0"


class YOLOAnnotationTool(QMainWindow):
    """带项目管理、彩色标签的YOLO自动标注工具主窗口"""

    def __init__(self):
        super().__init__()

        # 在__init__中定义所有实例属性
        self.current_project: Optional[Project] = None
        self.current_image_idx: int = -1
        self.current_process_idx: int = -1
        self.model: Optional[Any] = None
        self.filter_mode: str = "all"  # "all", "labeled", "unlabeled"
        logger.info("初始化YOLO自动标注工具窗口")

        # 初始化UI界面
        self.setWindowTitle("YOLO自动标注工具 - 无项目")
        self.setGeometry(100, 100, 1200, 800)

        # 设置窗口图标
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

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

        # 添加统计信息标签
        self.stats_label = QLabel("图片总数量：0 已标注数量：0 当前图片索引：0/0")
        self.config_layout.addRow("", self.stats_label)

        self.config_group.setLayout(self.config_layout)
        main_layout.addWidget(self.config_group)

        # 创建工作区域
        # 修正Qt.Horizontal的引用
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧图片列表
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.on_image_double_clicked)
        self.splitter.addWidget(self.image_list)

        # 添加筛选按钮
        self.filter_layout = QHBoxLayout()
        self.show_all_btn = QPushButton("全部图片")
        self.show_labeled_btn = QPushButton("已标注")
        self.show_unlabeled_btn = QPushButton("未标注")

        self.show_all_btn.clicked.connect(self.show_all_images)
        self.show_labeled_btn.clicked.connect(self.show_labeled_images)
        self.show_unlabeled_btn.clicked.connect(self.show_unlabeled_images)

        self.show_all_btn.setCheckable(True)
        self.show_labeled_btn.setCheckable(True)
        self.show_unlabeled_btn.setCheckable(True)

        # 默认选中全部图片
        self.show_all_btn.setChecked(True)

        self.filter_layout.addWidget(self.show_all_btn)
        self.filter_layout.addWidget(self.show_labeled_btn)
        self.filter_layout.addWidget(self.show_unlabeled_btn)

        # 将筛选按钮添加到主布局中
        main_layout.addLayout(self.filter_layout)

        # 右侧图片编辑区
        self.image_editor = ImageEditor()
        self.image_editor.annotation_updated.connect(self.on_annotation_updated)
        self.splitter.addWidget(self.image_editor)

        # 底部按钮区域
        self.btn_layout = QHBoxLayout()
        # 根据用户偏好，将主要操作按钮居中显示
        self.btn_layout.addStretch()

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)

        self.export_all_btn = QPushButton("导出所有结果")
        self.export_all_btn.clicked.connect(self.export_all_results)
        self.export_all_btn.setEnabled(False)

        self.btn_layout.addWidget(self.prev_btn)
        self.btn_layout.addWidget(self.next_btn)
        self.btn_layout.addWidget(self.export_all_btn)
        self.btn_layout.addStretch()

        # 添加到主布局
        main_layout.addWidget(self.splitter, 1)
        main_layout.addLayout(self.btn_layout)

        self.setCentralWidget(main_widget)

        # 初始禁用所有按钮，直到创建或打开项目
        self.set_widgets_enabled(False)

        # 筛选状态
        self.filter_mode = "all"  # "all", "labeled", "unlabeled"

    @property
    def current_image_path(self):
        """获取当前选中图片的路径"""
        if (not self.current_project or
                self.current_image_idx < 0 or
                self.current_image_idx >= len(self.current_project.image_paths)):
            return None
        return self.current_project.image_paths[self.current_image_idx]

    @property
    def filter_mode_description(self):
        """获取当前筛选模式的描述"""
        if self.filter_mode == "all":
            return "全部图片"
        elif self.filter_mode == "labeled":
            return "已标注"
        elif self.filter_mode == "unlabeled":
            return "未标注"
        return "未知"

    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        self.showMaximized()

    def show_all_images(self):
        """显示全部图片"""
        self.filter_mode = "all"
        self.show_all_btn.setChecked(True)
        self.show_labeled_btn.setChecked(False)
        self.show_unlabeled_btn.setChecked(False)
        self.update_image_list()

    def show_labeled_images(self):
        """只显示已标注的图片"""
        self.filter_mode = "labeled"
        self.show_all_btn.setChecked(False)
        self.show_labeled_btn.setChecked(True)
        self.show_unlabeled_btn.setChecked(False)
        self.update_image_list()

    def show_unlabeled_images(self):
        """只显示未标注的图片"""
        self.filter_mode = "unlabeled"
        self.show_all_btn.setChecked(False)
        self.show_labeled_btn.setChecked(False)
        self.show_unlabeled_btn.setChecked(True)
        self.update_image_list()

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # 修正Qt.Key_Return和Qt.Key_Enter的引用
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.complete_btn.isEnabled():
                self.complete_annotation()
                return  # 添加return避免继续传播事件
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
        self.manage_classes_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled and self.is_process_ready)
        self.export_all_btn.setEnabled((enabled and self.current_project.output_dir != "" and
                                        len(self.current_project.processed_images.keys()) > 0) if self.current_project is not None else False)
        self.prev_btn.setEnabled(enabled and self.current_image_idx > 0)
        self.next_btn.setEnabled(enabled and self.current_project and
                                 self.current_image_idx < len(self.current_project.image_paths) - 1)
        self.complete_btn.setEnabled(enabled and self.current_image_idx >= 0)

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

        if class_input:
            # 分割并处理每个标签（去空格、过滤空字符串）
            class_names = [cls.strip() for cls in class_input.split(',') if cls.strip()]
        else:
            class_names = default_classes.split(",")

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
        self.current_process_idx = self.current_project.last_processed_index
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

        # 重置筛选按钮状态
        self.filter_mode = "all"
        self.show_all_btn.setChecked(True)
        self.show_labeled_btn.setChecked(False)
        self.show_unlabeled_btn.setChecked(False)

        # 更新进度条和进度标签
        self.update_progress_on_open()

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
            self.current_process_idx = self.current_project.last_processed_index
            self.current_image_idx = self.current_project.last_processed_index
            self.update_project_info()
            self.update_image_list()
            self.image_editor.set_image(None)
            self.image_editor.set_annotations([])
            self.image_editor.set_class_info(
                self.current_project.class_names,
                self.current_project.class_colors
            )
            self.set_widgets_enabled(True)

            # 重置筛选按钮状态
            self.filter_mode = "all"
            self.show_all_btn.setChecked(True)
            self.show_labeled_btn.setChecked(False)
            self.show_unlabeled_btn.setChecked(False)

            # 更新配置显示
            self.image_dir_label.setText(self.current_project.image_dir)
            self.model_path_label.setText(self.current_project.model_path)
            self.output_dir_label.setText(self.current_project.output_dir)

            # 更新进度条和进度标签
            self.update_progress_on_open()

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
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            self.current_project.image_paths = []

            # 使用列表推导式提高性能
            try:
                files = os.listdir(dir_path)
                # 使用集合查找提高性能
                image_files = [
                    os.path.join(dir_path, file) for file in files
                    if os.path.splitext(file)[1].lower() in image_extensions
                ]
                self.current_project.image_paths = image_files
            except Exception as e:
                QMessageBox.warning(self, "错误", f"读取目录失败: {str(e)}")
                return

            # 更新图片列表
            self.update_image_list()

            self.check_process_ready()
            self.save_project()  # 自动保存项目

    def update_image_list(self) -> None:
        """更新图片列表，根据筛选模式显示并更新图片状态
        
        根据当前的filter_mode（全部、已标注或未标注）过滤图片列表，并更新每个图片项的状态
        包括颜色标记和选中状态。
        """
        if not self.current_project:
            self.image_list.clear()
            logger.debug("当前无项目，清空图片列表")
            return

        logger.debug(f"更新图片列表，筛选模式: {self.filter_mode_description}")

        # 获取当前选中项，以便更新后保持选中状态
        current_selected = None
        current_item = self.image_list.currentItem()
        if current_item:
            current_selected = current_item.data(Qt.ItemDataRole.UserRole)

        # 预先构建已处理图片的集合和标注状态字典，避免重复查询
        processed_images_set = set(self.current_project.processed_images.keys())
        # 使用缓存的_labeled_images集合优化查询性能
        labeled_images_set = getattr(self.current_project, '_labeled_images', set())

        # 根据筛选模式过滤图片
        filtered_paths: List[str] = []
        if self.filter_mode == "all":
            filtered_paths = self.current_project.image_paths
        elif self.filter_mode == "labeled":
            # 只显示已标注的图片
            filtered_paths = [path for path in self.current_project.image_paths
                              if path in labeled_images_set]
        elif self.filter_mode == "unlabeled":
            # 只显示未标注的图片
            filtered_paths = [path for path in self.current_project.image_paths
                              if path not in labeled_images_set]

        # 只在图片数量发生变化时才重新构建整个列表
        if self.image_list.count() != len(filtered_paths):
            logger.debug(f"图片数量变化，重新构建列表: {len(filtered_paths)} 张图片")
            self.image_list.clear()
            for path in filtered_paths:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.ItemDataRole.UserRole, path)
                # 设置工具提示，显示完整路径
                item.setToolTip(path)
                self.image_list.addItem(item)

        # 批量更新所有项目的颜色状态（只更新状态，不重建列表）
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_path = item.data(Qt.ItemDataRole.UserRole)
            # 标记有标注信息的图片为绿色，使用集合查询提高性能
            if image_path in labeled_images_set:
                item.setForeground(QColor(0, 128, 0))
                item.setText(f"{os.path.basename(image_path)}")
            else:
                item.setForeground(QColor(0, 0, 0))  # 默认黑色
                item.setText(os.path.basename(image_path))

        # 优化选中状态恢复逻辑
        if current_selected:
            # 优先尝试恢复之前选中的图片
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == current_selected:
                    self.image_list.setCurrentItem(item)
                    logger.debug(f"恢复选中的图片: {os.path.basename(current_selected)}")
                    break
        elif self.image_list.count() > 0 and self.current_image_idx >= 0:
            # 如果当前图片在筛选结果中，确保它被选中
            current_image_path = self.current_project.image_paths[self.current_image_idx]
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == current_image_path:
                    self.image_list.setCurrentItem(item)
                    logger.debug(f"选中当前图片索引: {self.current_image_idx}")
                    break

        # 更新进度标签
        self.update_progress_label()

    def update_progress_label(self):
        """更新进度标签显示"""
        if not self.current_project or not self.current_project.image_paths:
            self.stats_label.setText("图片总数量：0 已标注数量：0 当前图片索引：0/0")
            return

        total = self.current_project.total_count
        processed = self.current_project.processed_count
        current_index = self.current_image_idx + 1 if self.current_image_idx >= 0 else 0
        self.stats_label.setText(f"图片总数量：{total} 已标注数量：{processed} 当前图片索引：{current_index}/{total}")
        # 立即处理事件队列，确保UI及时更新
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

    def select_model_file(self) -> None:
        """选择并设置YOLO模型文件路径
        
        打开文件对话框让用户选择.pt格式的YOLO模型文件，更新项目设置并保存。
        """
        if not self.current_project:
            logger.warning("未创建或加载项目，无法选择模型文件")
            QMessageBox.warning(self, "警告", "请先创建或加载项目")
            return

        logger.info("打开模型文件选择对话框")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "YOLO模型文件 (*.pt);;所有文件 (*)")

        if file_path:
            try:
                # 验证文件是否存在
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"模型文件不存在: {file_path}")

                self.current_project.model_path = file_path
                self.model_path_label.setText(file_path)
                self.check_process_ready()
                self.save_project()  # 自动保存项目
                logger.info(f"成功设置模型文件: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"设置模型文件时出错: {str(e)}")
                QMessageBox.critical(self, "错误", f"设置模型文件失败: {str(e)}")

    def select_output_dir(self) -> None:
        """选择并设置标注输出目录
        
        打开目录选择对话框让用户选择保存标注结果的目录，更新项目设置并保存。
        """
        if not self.current_project:
            logger.warning("未创建或加载项目，无法选择输出目录")
            QMessageBox.warning(self, "警告", "请先创建或加载项目")
            return

        logger.info("打开输出目录选择对话框")
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")

        if dir_path:
            try:
                # 验证目录是否存在
                if not os.path.exists(dir_path):
                    raise FileNotFoundError(f"输出目录不存在: {dir_path}")

                self.current_project.output_dir = dir_path
                self.output_dir_label.setText(dir_path)
                has_processed = len(self.current_project.processed_images) > 0
                self.export_all_btn.setEnabled(has_processed)
                self.save_project()  # 自动保存项目
                logger.info(f"成功设置输出目录: {dir_path}")
            except Exception as e:
                logger.error(f"设置输出目录时出错: {str(e)}")
                QMessageBox.critical(self, "错误", f"设置输出目录失败: {str(e)}")

    @property
    def is_process_ready(self):
        """检查是否可以开始处理"""
        if not self.current_project:
            return False

        return self.current_project.is_ready

    def check_process_ready(self):
        """检查是否可以开始处理"""
        ready = self.is_process_ready
        self.process_btn.setEnabled(ready)
        return ready

    def start_processing(self):
        """开始处理图片（从当前位置开始）"""
        if not self.current_project or not self.is_process_ready:
            return

        self.set_widgets_enabled(False)
        self.current_process_idx = self.current_project.last_processed_index
        # 使用 QTimer.singleShot 避免递归调用导致的栈溢出
        QTimer.singleShot(1, self.process_next_image)  # 开始处理第一张

    def process_next_image(self) -> None:
        """处理下一张图片，实现批处理的核心逻辑
        
        检查处理进度，更新UI状态，处理单张图片，并在完成或出错时继续下一张。
        使用QTimer避免递归调用导致的栈溢出问题。
        """
        # 检查是否所有图片都已处理完成
        if self.current_process_idx >= len(self.current_project.image_paths):
            logger.info("所有图片处理完成，调用完成回调")
            self.on_processing_finished()
            return

        # 更新进度条
        total = len(self.current_project.image_paths)
        current = self.current_process_idx + 1
        # 实时更新进度条
        progress = int((self.current_process_idx / total) * 100)
        self.progress_bar.setValue(progress)

        # 添加进度信息到状态栏
        status_text = f"正在处理 {current}/{total} ({progress}%)"
        self.statusBar().showMessage(status_text)

        # 实时更新进度标签
        self.update_progress_label()

        # 获取当前要处理的图片路径
        current_image_path = self.current_project.image_paths[self.current_process_idx]
        current_image_name = os.path.basename(current_image_path)

        logger.info(f"处理图片 {current}/{total}: {current_image_name}")

        # 如果已处理过则跳过
        if current_image_path in self.current_project.processed_images:
            logger.debug(f"图片已处理，跳过: {current_image_name}")
            self.current_process_idx += 1
            # 使用单次定时器而不是递归调用，避免栈溢出
            QTimer.singleShot(1, self.process_next_image)
            return

        try:
            # 直接在主线程处理图片
            image, annotations = self.process_single_image(current_image_path)

            # 仅当成功处理且有有效标注时才调用回调
            if image is not None:
                logger.debug(f"成功处理图片，准备回调: {current_image_name}")
                self.on_single_image_processed(current_image_path, image, annotations)
            else:
                logger.error(f"图片处理返回空结果: {current_image_name}")
                raise Exception("图片处理返回空结果")

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            logger.error(f"图片 {current_image_path} 处理错误: {error_msg}")

            # 避免频繁弹窗，只在处理少量图片或重要错误时显示
            if self.current_process_idx < 10 or (self.current_process_idx % 50 == 0):
                QMessageBox.warning(self, "处理错误", f"处理 {current_image_name} 时出错:\n{error_msg}")

            # 记录错误到错误日志
            if hasattr(self, 'error_log'):
                self.error_log.append((current_image_path, str(e)))

            # 继续处理下一张图片
            self.current_process_idx += 1
            QTimer.singleShot(1, self.process_next_image)

    def process_single_image(self, image_path: str) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """处理单张图片的核心逻辑
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            tuple: (处理后的图片数组, 标注列表)
            
        Raises:
            Exception: 模型加载或推理失败时抛出
        """
        # 初始化模型（确保只初始化一次）
        if not hasattr(self, 'model') or self.model is None:
            try:
                logger.info(f"加载YOLO模型: {self.current_project.model_path}")
                self.model = YOLO(self.current_project.model_path)
                # 设置设备优先级: CUDA > MPS > CPU
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                self.model.to(device=device)
                logger.info(f"模型加载成功，使用设备: {device}")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise Exception(f"模型加载失败: {str(e)}")

        # 处理图片
        logger.debug(f"读取图片: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图片文件: {image_path}")
            raise Exception(f"无法读取图片文件: {os.path.basename(image_path)}")

        try:
            # 使用优化的模型推理参数
            logger.debug(f"执行模型推理: {os.path.basename(image_path)}")
            results = self.model(image, verbose=False, batch=1, device="cuda" if torch.cuda.is_available() else "cpu")
            logger.debug(f"模型推理完成，检测到 {sum(len(result.boxes) for result in results)} 个对象")
        except Exception as e:
            logger.error(f"模型推理失败: {str(e)}")
            raise Exception(f"模型推理失败: {str(e)}")

        annotations: List[Dict[str, Any]] = []
        # 创建类名字典以提高查找性能
        class_names_set = set(self.current_project.class_names)
        # 添加置信度阈值配置
        confidence_threshold = getattr(self.current_project, 'confidence_threshold', 0.25)

        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                boxes_np = boxes.cpu().numpy()
                model_names = result.names
                for box in boxes_np:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    # 根据置信度阈值过滤低置信度检测
                    if conf < confidence_threshold:
                        continue

                    cls = int(box.cls[0])
                    class_name = model_names[cls]
                    # 使用集合查找提高性能
                    if class_name in class_names_set:
                        annotations.append({
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": float(conf),
                            "class_id": int(cls),
                            "class": class_name
                        })

        # 释放CUDA内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA缓存已清理")

        logger.info(f"图片处理完成: {os.path.basename(image_path)}, 检测到 {len(annotations)} 个有效标注")
        return image, annotations

    def update_single_item_in_list(self, image_path):
        """更新列表中单个图片项的状态，支持筛选模式"""
        # 检查当前筛选模式是否应该显示此图片
        should_show = True
        processed_images_set = set(self.current_project.processed_images.keys())

        if self.filter_mode == "labeled":
            # 只显示已标注的图片
            should_show = image_path in processed_images_set and self.current_project.has_annotations(image_path)
        elif self.filter_mode == "unlabeled":
            # 只显示未标注的图片
            should_show = not (image_path in processed_images_set and self.current_project.has_annotations(image_path))

        # 查找现有项
        item_index = -1
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path == image_path:
                item_index = i
                break

        # 如果应该显示但未找到，添加到列表
        if should_show and item_index == -1:
            item = QListWidgetItem(os.path.basename(image_path))
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self.image_list.addItem(item)
            item_index = self.image_list.count() - 1

        # 如果不应该显示但找到了，从列表中移除
        elif not should_show and item_index != -1:
            self.image_list.takeItem(item_index)
            return

        # 如果应该显示且找到了，更新状态
        if should_show and item_index != -1:
            item = self.image_list.item(item_index)
            # 重置颜色
            item.setForeground(QColor(0, 0, 0))  # 默认黑色

            # 标记有标注信息的图片为绿色
            if image_path in processed_images_set and self.current_project.has_annotations(image_path):
                # 有标注信息 - 绿色
                item.setForeground(QColor(0, 128, 0))

    def on_single_image_processed(self, image_path: str, image: np.ndarray, annotations: List[Dict[str, Any]]) -> None:
        """单张图片处理完成回调
        
        Args:
            image_path: 图片文件路径
            image: 处理后的图片数组
            annotations: 检测到的标注列表
        """
        try:
            # 更新项目状态
            self.current_project.processed_images[image_path] = (image, annotations)
            self.current_project.last_processed_index = self.current_process_idx

            # 更新已标注图片缓存集合，优化后续查询性能
            if not hasattr(self.current_project, '_labeled_images'):
                self.current_project._labeled_images = set()
            self.current_project._labeled_images.add(image_path)

            logger.debug(f"更新项目状态: {os.path.basename(image_path)}, 标注数量: {len(annotations)}")

            # 更新单个列表项状态
            self.update_single_item_in_list(image_path)

            # 更新进度标签
            self.update_progress_label()

            # 显示当前处理的图片
            self.current_image_idx = self.current_process_idx
            self.show_current_image()

            # 自动保存项目（每处理10张或最后一张时）
            total = len(self.current_project.image_paths)
            if (self.current_process_idx + 1) % 10 == 0 or (self.current_process_idx + 1) == total:
                self.save_project()
                logger.info(f"自动保存项目进度: {self.current_process_idx + 1}/{total}")

            # 恢复UI交互
            self.set_widgets_enabled(True)
            self.complete_btn.setEnabled(True)

            # 继续处理下一张，使用定时器避免递归调用导致的栈溢出
            self.current_process_idx += 1
            QTimer.singleShot(1, self.process_next_image)

        except Exception as e:
            error_msg = f"处理回调失败: {str(e)}"
            logger.error(f"图片 {image_path} 处理回调错误: {error_msg}")
            QMessageBox.critical(self, "处理回调错误", f"处理 {os.path.basename(image_path)} 回调时出错:\n{error_msg}")

            # 确保即使出错也能继续处理
            self.current_process_idx += 1
            QTimer.singleShot(1, self.process_next_image)

    def on_process_error(self, image_path, error_msg):
        """处理错误回调"""
        print(f"图片 {image_path} 处理错误: {error_msg}")
        QMessageBox.warning(self, "处理错误", f"处理 {os.path.basename(image_path)} 时出错:\n{error_msg}")
        self.current_process_idx += 1
        self.process_next_image()  # 继续处理下一张

    def update_progress(self, value):
        """更新进度条"""
        # 计算全局进度
        start_idx = self.current_project.last_processed_index
        total = len(self.current_project.image_paths)
        processed = len(self.current_project.image_paths) - len(self.current_project.image_paths[start_idx:])
        current = int(processed / total * 100) + int(
            value * len(self.current_project.image_paths[start_idx:]) / total / 100)
        self.progress_bar.setValue(current)

    def on_image_processed(self, image_path, annotations):
        """处理完一张图片后的回调"""
        if not self.current_project:
            return

        self.current_project.processed_images[image_path] = (None, annotations)

        # 更新图片列表状态
        self.update_image_list()

        # 获取当前图片索引
        if image_path in self.current_project.image_paths:
            self.current_image_idx = self.current_project.image_paths.index(image_path)
            self.current_project.last_processed_index = self.current_image_idx

            # 显示当前图片
            self.show_current_image()
            self.update_nav_buttons()
            self.set_widgets_enabled(True)
            self.complete_btn.setEnabled(True)

    def on_processing_finished(self):
        """所有图片处理完成后的回调"""
        self.progress_bar.setValue(100)
        # 更新进度标签
        self.update_progress_label()
        # 启用按钮
        self.set_widgets_enabled(True)
        self.save_project()  # 保存处理结果

        if len(self.current_project.processed_images) > 0:
            QMessageBox.information(self, "完成", "所有图片处理完成")
            # 如果还没有显示任何图片，显示第一张
            if self.current_image_idx == -1 and len(self.current_project.image_paths) > 0:
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

        # 如果当前图片没有标注信息，则自动进行推理处理
        if not self.current_project.has_annotations(image_path):
            try:
                # 进行推理处理
                image, annotations = self.process_single_image(image_path)
                # 更新项目状态
                self.current_project.processed_images[image_path] = (image, annotations)

                # 更新列表项状态
                self.update_single_item_in_list(image_path)
            except Exception as e:
                error_msg = f"处理失败: {str(e)}"
                print(f"图片 {image_path} 处理错误: {error_msg}")
                QMessageBox.warning(self, "处理错误", f"处理 {os.path.basename(image_path)} 时出错:\n{error_msg}")
                self.image_editor.set_image(None)
                self.image_editor.set_annotations([])
                return

        # 显示图片和标注
        if image_path in self.current_project.processed_images:
            image, annotations = self.current_project.processed_images[image_path]

            # 如果图像为None（从项目加载的情况），重新加载
            if image is None:
                image = cv2.imread(image_path)
                # 仅在图像成功加载时更新缓存
                if image is not None:
                    self.current_project.processed_images[image_path] = (image, annotations)

            if image is not None:
                self.image_editor.set_image(image)
                self.image_editor.set_annotations(annotations)
            else:
                # 图像加载失败的处理
                QMessageBox.warning(self, "错误", f"无法加载图片: {image_path}")
                self.image_editor.set_image(None)
                self.image_editor.set_annotations([])

            # 高亮显示列表中的当前项
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == image_path:
                    self.image_list.setCurrentItem(item)
                    break
            self.image_editor.update()

        else:
            # 图像不在processed_images中
            self.image_editor.set_image(None)
            self.image_editor.set_annotations([])

        # 更新当前图片索引显示
        self.update_progress_label()

    def on_image_double_clicked(self, item):
        """双击图片列表项"""
        if not self.current_project:
            return

        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path in self.current_project.image_paths:
            self.current_image_idx = self.current_project.image_paths.index(image_path)
            self.show_current_image()
            self.update_nav_buttons()

    def next_image(self):
        """显示下一张图片"""
        if (self.current_project and
                self.current_image_idx < len(self.current_project.image_paths) - 1):
            self.current_image_idx += 1
            self.show_current_image()
            self.update_nav_buttons()

    def prev_image(self):
        """显示上一张图片"""
        if self.current_project and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.show_current_image()
            self.update_nav_buttons()

    def update_nav_buttons(self):
        """更新导航按钮状态"""
        if not self.current_project:
            return

        self.prev_btn.setEnabled(self.current_image_idx > 0)
        self.next_btn.setEnabled(self.current_image_idx < len(self.current_project.image_paths) - 1)
        self.complete_btn.setEnabled(True)

    def complete_annotation(self):
        """完成当前图片标注，进入下一张"""
        if (not self.current_project or
                self.current_image_idx < 0 or
                self.current_image_idx >= len(self.current_project.image_paths)):
            return

        # 移动到下一张
        if self.current_image_idx < len(self.current_project.image_paths) - 1:
            self.current_image_idx += 1
            self.show_current_image()
            self.update_nav_buttons()
        else:
            QMessageBox.information(self, "完成", "所有图片已处理完毕")
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
            # 更新列表项状态
            self.update_single_item_in_list(image_path)
            # 只在非拖拽操作时保存项目
            if not self.image_editor.during_drag_operation:
                self.save_project()  # 自动保存修改

    def export_all_results(self) -> None:
        """导出所有结果到YOLO格式数据集目录结构
        
        显示数据集划分对话框，根据用户设置的比例将数据分为训练集、验证集和测试集，
        并导出为标准YOLO格式（images和labels目录结构）。
        """
        try:
            if not self.current_project:
                logger.warning("无当前项目，无法导出结果")
                QMessageBox.warning(self, "警告", "请先创建或加载项目")
                return

            if not self.current_project.processed_images:
                logger.warning("没有已处理的图片，无法导出结果")
                QMessageBox.warning(self, "警告", "没有已处理的图片")
                return

            if not self.current_project.output_dir:
                logger.warning("未设置输出目录，无法导出结果")
                QMessageBox.warning(self, "警告", "请先设置输出目录")
                return

            logger.info("开始导出所有结果")

            # 显示比例设置对话框
            dialog = DatasetSplitDialog()
            if dialog.exec() != QDialog.DialogCode.Accepted:
                logger.info("用户取消了数据集划分设置")
                return

            ratios = dialog.get_ratios()
            if not ratios:
                logger.warning("未获取到有效的数据集划分比例")
                return

            train_ratio, val_ratio, test_ratio = ratios
            logger.info(f"数据集划分比例 - 训练: {train_ratio}, 验证: {val_ratio}, 测试: {test_ratio}")

            # 创建目录结构
            unlabeled_dir = os.path.join(self.current_project.output_dir, "unlabeled")
            base_dir = os.path.join(self.current_project.output_dir, "labeled")

            # 有标注数据的目录结构 (images存放图片, labels存放标注文件)
            train_img_dir = os.path.join(base_dir, "train", "images")
            train_label_dir = os.path.join(base_dir, "train", "labels")
            val_img_dir = os.path.join(base_dir, "val", "images")
            val_label_dir = os.path.join(base_dir, "val", "labels")
            test_img_dir = os.path.join(base_dir, "test", "images")
            test_label_dir = os.path.join(base_dir, "test", "labels")

            # 使用utils.ensure_directory确保目录存在，提高安全性
            from utils import ensure_directory
            dirs_to_create = [
                unlabeled_dir,
                train_img_dir, train_label_dir,
                val_img_dir, val_label_dir,
                test_img_dir, test_label_dir
            ]

            for dir_path in dirs_to_create:
                ensure_directory(dir_path)
                logger.debug(f"确保目录存在: {dir_path}")

            # 分离有标注和无标注图片
            labeled_images: List[str] = []
            unlabeled_images: List[str] = []

            for image_path in self.current_project.image_paths:
                # 检查是否有标注信息且标注不为空
                if self.current_project.has_annotations(image_path):
                    labeled_images.append(image_path)
                else:
                    unlabeled_images.append(image_path)

            logger.info(f"找到 {len(labeled_images)} 张有标注图片，{len(unlabeled_images)} 张无标注图片")

            # 随机打乱已标注图片顺序
            random.shuffle(labeled_images)
            total_labeled = len(labeled_images)

            # 计算各数据集数量
            train_count = int(total_labeled * train_ratio)
            val_count = int(total_labeled * val_ratio)

            # 分配图片到各个数据集
            train_images = labeled_images[:train_count]
            val_images = labeled_images[train_count:train_count + val_count]
            test_images = labeled_images[train_count + val_count:]

            logger.info(
                f"数据集划分完成 - 训练集: {len(train_images)}, 验证集: {len(val_images)}, 测试集: {len(test_images)}")

            # 导出已标注图片及对应的标注文件
            datasets = [
                (train_images, train_img_dir, train_label_dir),
                (val_images, val_img_dir, val_label_dir),
                (test_images, test_img_dir, test_label_dir)
            ]

            total = len(labeled_images) + len(unlabeled_images)
            current = 0

            for images, img_dir, label_dir in datasets:
                for image_path in images:
                    # 复制原始图片到对应images目录
                    img_filename = os.path.basename(image_path)
                    shutil.copy2(image_path, os.path.join(img_dir, img_filename))

                    # 生成并保存标注文件到对应labels目录
                    self.export_annotation_file(image_path, label_dir)

                    current += 1
                    # 实时更新进度条
                    self.progress_bar.setValue(int(current / total * 100))

            # 导出未标注图片
            for image_path in unlabeled_images:
                img_filename = os.path.basename(image_path)
                shutil.copy2(image_path, os.path.join(unlabeled_dir, img_filename))
                current += 1
                # 实时更新进度条
                self.progress_bar.setValue(int(current / total * 100))

            # 预先创建类别名称到ID的映射以提高查找性能
            class_name_to_id = {name: i for i, name in enumerate(self.current_project.class_names)}

            # 添加导出训练集配置文件
            self._export_dataset_config(base_dir, class_name_to_id)

            QMessageBox.information(None, "完成",
                                    f"所有图片已导出到 {base_dir}\n"
                                    f"已标注: {len(labeled_images)} 张 (训练集: {len(train_images)}, "
                                    f"验证集: {len(val_images)}, 测试集: {len(test_images)})\n"
                                    f"未标注: {len(unlabeled_images)} 张")

        except Exception as e:
            logger.error(f"导出结果时出错: {str(e)}")
            QMessageBox.critical(self, "导出错误", f"导出结果失败: {str(e)}")

    @staticmethod
    def _export_dataset_config(base_dir: str, class_name_to_id: Dict[str, int]) -> None:
        """导出数据集配置文件
        
        Args:
            base_dir: 数据集基础目录
            class_name_to_id: 类别名称到ID的映射
        """
        try:
            # 导出classes.txt文件
            classes_file = os.path.join(base_dir, "classes.txt")
            with open(classes_file, "w", encoding="utf-8") as f:
                for class_name in sorted(class_name_to_id.keys(), key=lambda x: class_name_to_id[x]):
                    f.write(f"{class_name}\n")
            logger.info(f"成功导出类别配置文件: {classes_file}")

            # 导出dataset.yaml配置文件（YOLO格式）
            yaml_file = os.path.join(base_dir, "dataset.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("train: train/images\n")
                f.write("val: val/images\n")
                f.write("test: test/images\n\n")
                f.write(f"nc: {len(class_name_to_id)}\n")
                f.write("names: [")
                f.write(", ".join(
                    [f'\"{name}\"' for name in sorted(class_name_to_id.keys(), key=lambda x: class_name_to_id[x])]))
                f.write("]\n")
            logger.info(f"成功导出YOLO配置文件: {yaml_file}")

        except Exception as e:
            logger.error(f"导出配置文件时出错: {str(e)}")

    def export_annotation_file(self, image_path: str, label_dir: str) -> bool:
        """导出单个图片的YOLO格式标注文件
        
        Args:
            image_path: 图像文件路径
            label_dir: 标注文件保存目录
            
        Returns:
            bool: 导出成功返回True，失败返回False
        """
        try:
            if not self.current_project:
                logger.error("当前无活动项目")
                return False

            if image_path not in self.current_project.processed_images:
                logger.warning(f"图片未处理，跳过导出: {image_path}")
                return False

            # 获取标注信息
            image, annotations = self.current_project.processed_images[image_path]
            if annotations is None:
                logger.debug(f"图片无标注信息: {image_path}")
                return False

            if not annotations:
                logger.debug(f"图片标注为空: {image_path}")
                # 创建空文件表明已处理但无标注
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_txt_path = os.path.join(label_dir, f"{base_name}.txt")
                open(output_txt_path, 'w').close()  # 创建空文件
                logger.debug(f"创建空标注文件: {output_txt_path}")
                return True

            # 如果图像为None，尝试加载它
            if image is None:
                logger.debug(f"缓存中无图像数据，尝试加载: {image_path}")
                image = cv2.imread(image_path)
                # 更新缓存中的图像
                if image is not None:
                    self.current_project.processed_images[image_path] = (image, annotations)
                    logger.debug(f"成功加载并更新图像缓存: {image_path}")

            # 获取图片尺寸用于坐标归一化
            if image is not None:
                height, width = image.shape[:2]
                logger.debug(f"图像尺寸: {width}x{height}")
            else:
                # 如果无法加载图像，给出警告并跳过
                logger.warning(f"无法加载图像，跳过导出: {image_path}")
                return False

            # 验证目录存在
            if not os.path.exists(label_dir):
                try:
                    os.makedirs(label_dir, exist_ok=True)
                    logger.info(f"创建标注目录: {label_dir}")
                except Exception as e:
                    logger.error(f"创建标注目录失败: {str(e)}")
                    return False

            # 生成标注文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_txt_path = os.path.join(label_dir, f"{base_name}.txt")

            # 创建类别名称到ID的映射以提高查找性能
            class_name_to_id = {name: i for i, name in enumerate(self.current_project.class_names)}

            # 写入YOLO格式标注
            with open(output_txt_path, "w", encoding="utf-8") as f:
                valid_annotations = 0
                for annot in annotations:
                    try:
                        if "box" not in annot or "class" not in annot:
                            logger.warning(f"标注数据不完整: {annot}")
                            continue

                        x1, y1, x2, y2 = annot["box"]

                        # 验证坐标有效性
                        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                            logger.warning(f"标注坐标超出图像范围: {image_path}, 坐标: {x1}, {y1}, {x2}, {y2}")
                            # 裁剪坐标到有效范围
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(width, x2)
                            y2 = min(height, y2)

                        # 检查边界框是否有效
                        if x2 <= x1 or y2 <= y1:
                            logger.warning(f"无效边界框(宽或高为0): {image_path}, 坐标: {x1}, {y1}, {x2}, {y2}")
                            continue

                        # 转换为YOLO格式：中心点坐标和宽高（归一化）
                        cx = (x1 + x2) / 2 / width
                        cy = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height

                        # 确保坐标在有效范围内
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w = max(0.0, min(1.0, w))
                        h = max(0.0, min(1.0, h))

                        # 获取类别ID
                        class_name = annot["class"]
                        if class_name not in class_name_to_id:
                            logger.error(f"未知类别: {class_name}")
                            continue

                        class_id = class_name_to_id[class_name]

                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        valid_annotations += 1
                    except Exception as e:
                        logger.error(f"处理单个标注时出错: {str(e)}")
                        continue

                # 强制刷新文件缓冲区确保数据写入
                f.flush()
                os.fsync(f.fileno())

            if valid_annotations > 0:
                logger.info(f"成功导出 {valid_annotations} 个标注到: {output_txt_path}")
                return True
            else:
                # 如果没有有效标注，删除空文件
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                logger.warning(f"无有效标注导出: {image_path}")
                return False

        except Exception as e:
            logger.error(f"导出标注文件时出错: {str(e)}")
            return False

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", "YOLO自动标注工具\n版本 1.2\n支持项目管理、自定义标签和彩色标注")

    def closeEvent(self, event):
        """窗口关闭事件"""
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

    def update_progress_on_open(self):
        """在打开项目时更新进度条和进度标签"""
        if not self.current_project:
            return

        # 更新进度条
        self.progress_bar.setValue(self.current_project.progress)

        # 更新进度标签
        self.update_progress_label()

        # 重置筛选按钮状态
        self.filter_mode = "all"
        self.show_all_btn.setChecked(True)
        self.show_labeled_btn.setChecked(False)
        self.show_unlabeled_btn.setChecked(False)
