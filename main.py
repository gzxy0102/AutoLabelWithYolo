import sys
from PySide6.QtWidgets import QApplication
from ui import YOLOAnnotationTool

def main():
    app = QApplication(sys.argv)
    # 设置应用程序样式
    app.setStyle("Fusion")
    window = YOLOAnnotationTool()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
