import sys
import os
import logging
from typing import Optional
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from ui import YOLOAnnotationTool

# 配置根日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'auto_label_tool.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_app_icon(app: QApplication, window: Optional[YOLOAnnotationTool] = None) -> None:
    """设置应用程序和窗口图标
    
    Args:
        app: QApplication实例
        window: YOLOAnnotationTool窗口实例（可选）
    """
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if os.path.exists(icon_path):
        try:
            icon = QIcon(icon_path)
            app.setWindowIcon(icon)
            if window:
                window.setWindowIcon(icon)
            logger.info("成功设置应用程序图标")
        except Exception as e:
            logger.warning(f"设置图标时出错: {str(e)}")
    else:
        logger.warning(f"未找到图标文件: {icon_path}")

def main() -> None:
    """应用程序主入口"""
    try:
        # 创建应用程序实例
        app = QApplication(sys.argv)

        # 设置应用程序样式
        app.setStyle("Fusion")

        # 设置应用程序属性
        app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)

        # 创建主窗口
        logger.info("初始化YOLO标注工具...")
        window = YOLOAnnotationTool()

        # 设置图标
        setup_app_icon(app, window)

        # 显示窗口
        window.show()
        logger.info("应用程序启动成功")

        # 运行应用程序
        sys.exit(app.exec())

    except Exception as e:
        logger.critical(f"应用程序启动失败: {str(e)}", exc_info=True)
        # 显示错误对话框
        from PySide6.QtWidgets import QMessageBox
        error_app = QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "启动错误",
            f"应用程序启动失败:\n{str(e)}",
            QMessageBox.StandardButton.Ok
        )
        sys.exit(1)

if __name__ == "__main__":
    main()