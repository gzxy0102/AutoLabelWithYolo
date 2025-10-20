import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui import YOLOAnnotationTool

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序图标
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    window = YOLOAnnotationTool()
    
    # 为窗口设置图标
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()