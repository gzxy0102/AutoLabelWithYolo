from PySide6.QtGui import (QColor)
from PySide6.QtWidgets import (QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox,
                               QInputDialog, QFormLayout,
                               QDialog,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QColorDialog, QLineEdit, QDialogButtonBox)

from utils import generate_distinct_colors


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
        if 0 <= current_row < len(self.class_names):
            old_name = self.class_names[current_row]
            new_name, ok = QInputDialog.getText(
                self, "编辑标签", "请输入新的标签名称:", text=old_name)
            if ok and new_name and new_name not in self.class_names:
                self.class_names[current_row] = new_name
                self.update_table()

    def change_color(self):
        """修改选中标签的颜色"""
        current_row = self.class_table.currentRow()
        if 0 <= current_row < len(self.class_names):
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
        if 0 <= current_row < len(self.class_names):
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