import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt, QPoint, QTimer, QRect
from PySide6.QtGui import (QPixmap, QPainter, QPen, QKeySequence,
                           QShortcut, QLinearGradient, QColor, QFont)


class ImageCompareWidget(QWidget):
    STYLE_CONFIG = {
        "bg_gradient": (QColor(10, 20, 30), QColor(20, 30, 50)),
        "grid_color": QColor(0, 255, 255, 30),
        "neon_blue": QColor(0, 168, 255),
        "neon_blue1": QColor(0, 0, 255),
        "neon_white": QColor(255, 255, 255),
        "panel_bg": QColor(0, 0, 30, 200),
        "text_color": QColor(200, 220, 255),
        "glow_speed": 4
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image1 = QPixmap()
        self._image2 = QPixmap()
        self.mode = 0
        self.split_pos = 0.5
        self._dragging = False
        self._cached_scale = 1.0
        self._cached_offset = QPoint(0, 0)
        self._scaled_size = QPoint(0, 0)

        # 动画控制
        self._auto_animate = False
        self._animation_step = 0.005  # 0.005
        self._animation_dir = 1
        self._setup_timers()
        self._init_shortcuts()
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)

    def _setup_timers(self):
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._update_glow)
        self._glow_timer.start(50)

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._auto_update_split)
        self._anim_timer.start(30)

    def _init_shortcuts(self):
        shortcuts = (("H", 0), ("V", 1), ("A", 2))
        for key, mode in shortcuts:
            QShortcut(QKeySequence(key), self, lambda m=mode: self.set_mode(m))

    def set_images(self, image1: QPixmap, image2: QPixmap):
        if image1.isNull() or image2.isNull():
            QMessageBox.critical(self, "错误", "图片加载失败！")
            return

        self._image1, self._image2 = image1, image2
        if image1.size() != image2.size():
            self._unify_sizes()
        self._cache_scale_info()
        self.update()

    def _unify_sizes(self):
        target_size = self._image1.size()
        scaled = self._image2.scaled(
            target_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        self._image2 = QPixmap(target_size)
        self._image2.fill(Qt.black)
        painter = QPainter(self._image2)
        painter.drawPixmap(
            (target_size.width() - scaled.width()) // 2,
            (target_size.height() - scaled.height()) // 2,
            scaled
        )
        painter.end()

    def _cache_scale_info(self):
        if self._image1.isNull():
            return

        img_w, img_h = self._image1.width(), self._image1.height()
        win_w, win_h = self.width(), self.height()

        scale = min(win_w / img_w, win_h / img_h) if img_w * img_h > 0 else 1.0
        self._cached_scale = scale
        self._scaled_size = QPoint(
            int(img_w * scale),
            int(img_h * scale)
        )
        self._cached_offset = QPoint(
            (win_w - self._scaled_size.x()) // 2,
            (win_h - self._scaled_size.y()) // 2
        )

    def set_mode(self, mode: int):
        self.mode = max(0, min(mode, 2))
        self.update()

    def paintEvent(self, event):
        if self._image1.isNull() or self._image2.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self._draw_background(painter)
        self._draw_content(painter)
        self._draw_info_panel(painter)

    def _draw_background(self, painter: QPainter):
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, self.STYLE_CONFIG["bg_gradient"][0])
        gradient.setColorAt(1, self.STYLE_CONFIG["bg_gradient"][1])
        painter.fillRect(self.rect(), gradient)

        painter.setPen(QPen(self.STYLE_CONFIG["grid_color"], 1, Qt.DotLine))
        for x in range(0, self.width(), 40):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 40):
            painter.drawLine(0, y, self.width(), y)

    def _draw_content(self, painter: QPainter):
        x, y = self._cached_offset.x(), self._cached_offset.y()
        scaled_w, scaled_h = self._scaled_size.x(), self._scaled_size.y()

        if self.mode == 0:
            split_x = x + int(scaled_w * self.split_pos)
            painter.drawPixmap(x, y, self._image1.scaled(
                scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ))
            painter.setClipRect(split_x, y, scaled_w - split_x + x, scaled_h)
            painter.drawPixmap(x, y, self._image2.scaled(
                scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ))
            painter.setClipping(False)
            self._draw_split_line(painter, QRect(split_x - 1, y, 2, scaled_h), 'vertical')

        elif self.mode == 1:
            split_y = y + int(scaled_h * self.split_pos)
            painter.drawPixmap(x, y, self._image1.scaled(
                scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ))
            painter.setClipRect(x, split_y, scaled_w, scaled_h - split_y + y)
            painter.drawPixmap(x, y, self._image2.scaled(
                scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ))
            painter.setClipping(False)
            self._draw_split_line(painter, QRect(x, split_y - 1, scaled_w, 2), 'horizontal')

        else:
            painter.setOpacity(1 - self.split_pos)
            painter.drawPixmap(x, y, self._image1.scaled(
                scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ))
            painter.setOpacity(self.split_pos)
            painter.drawPixmap(x, y, self._image2.scaled(
                scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ))

    def _draw_split_line(self, painter: QPainter, rect: QRect, direction: str):
        # painter.setPen(QPen(self.STYLE_CONFIG["neon_blue"], 3, Qt.SolidLine, Qt.RoundCap))

        # 修改画笔样式为虚线
        pen = QPen(self.STYLE_CONFIG["neon_white"], 3, Qt.DashLine, Qt.RoundCap)
        # pen = QPen(self.STYLE_CONFIG["neon_blue"], 3, Qt.DashLine, Qt.RoundCap)
        pen.setDashPattern([4, 4])  # 4像素实线 + 4像素空白
        painter.setPen(pen)

        if direction == 'vertical':
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
        else:
            painter.drawLine(rect.topLeft(), rect.topRight())

    def _draw_info_panel(self, painter: QPainter):
        """绘制信息面板"""
        panel_rect = QRect(20, 20, 200, 90)
        painter.setBrush(self.STYLE_CONFIG["panel_bg"])
        painter.setPen(QPen(self.STYLE_CONFIG["neon_blue"], 1))
        painter.drawRoundedRect(panel_rect, 5, 5)

        # 保留原始文字格式
        mode_names = ["水平分屏 (H)", "垂直分屏 (V)", "透明度混合 (A)"]
        info_text = (
            f"模式: {mode_names[self.mode]}\n"
            f"分屏比例: {self.split_pos * 100:.1f}%\n"
            f"分辨率: {self._scaled_size.x()}x{self._scaled_size.y()}"
        )

        original_font = painter.font()
        new_font = QFont("Times New Roman", 12)
        new_font.setPixelSize(20)
        painter.setFont(new_font)
        painter.setPen(self.STYLE_CONFIG["text_color"])

        text_rect = panel_rect.adjusted(10, 10, -10, -10)
        painter.drawText(text_rect, Qt.TextWordWrap, info_text)
        painter.setFont(original_font)

    def resizeEvent(self, event):
        self._cache_scale_info()
        self.update()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._image1.isNull():
            self._dragging = True
            self._update_split_pos(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if self._dragging and not self._image1.isNull():
            self._update_split_pos(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def _update_split_pos(self, pos):
        win_w, win_h = self.width(), self.height()
        if not (0 <= pos.x() <= win_w and 0 <= pos.y() <= win_h):
            self._auto_animate = True
            return

        x, y = pos.x(), pos.y()
        scaled_w, scaled_h = self._scaled_size.x(), self._scaled_size.y()
        x_offset, y_offset = self._cached_offset.x(), self._cached_offset.y()

        if self.mode == 0:
            relative_x = max(0, min(x - x_offset, scaled_w))
            self.split_pos = relative_x / scaled_w if scaled_w else 0.5
        elif self.mode == 1:
            relative_y = max(0, min(y - y_offset, scaled_h))
            self.split_pos = relative_y / scaled_h if scaled_h else 0.5
        else:
            self.split_pos = max(0.0, min(1.0, x / win_w))

        self.split_pos = max(0.0, min(1.0, self.split_pos))
        self.update()

    def enterEvent(self, event):
        self._auto_animate = False
        self._anim_timer.stop()

    def leaveEvent(self, event):
        self._auto_animate = True
        self._anim_timer.start(30)

    def _auto_update_split(self):
        if not self._auto_animate:
            return

        self.split_pos += self._animation_step * self._animation_dir
        if self.split_pos >= 1 or self.split_pos <= 0:
            self._animation_dir *= -1
            self.split_pos = max(0.0, min(1.0, self.split_pos))
        self.update()

    def _update_glow(self):
        pass  # 可根据需要实现呼吸灯效果


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Imgsli Clone - PySide6")
        self.setGeometry(100, 100, 1024, 768)
        self.compare_widget = ImageCompareWidget()
        # self.compare_widget = QWidget()
        self.setCentralWidget(self.compare_widget)

        toolbar = self.addToolBar("Tools")
        btn_open = QPushButton("打开图片")
        btn_open.clicked.connect(self.open_images)
        toolbar.addWidget(btn_open)

        btn_mode_h = QPushButton("水平模式 (H)")
        btn_mode_h.clicked.connect(lambda: self.compare_widget.set_mode(0))
        toolbar.addWidget(btn_mode_h)

        btn_mode_v = QPushButton("垂直模式 (V)")
        btn_mode_v.clicked.connect(lambda: self.compare_widget.set_mode(1))
        toolbar.addWidget(btn_mode_v)

        btn_mode_a = QPushButton("混合模式 (A)")
        btn_mode_a.clicked.connect(lambda: self.compare_widget.set_mode(2))
        toolbar.addWidget(btn_mode_a)

    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "",
                                                "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        if len(paths) >= 2:
            orig_pixmap = QPixmap(paths[0])
            processed_pixmap = QPixmap(paths[1])
            self.compare_widget.set_images(orig_pixmap, processed_pixmap)
        else:
            QMessageBox.warning(self, "警告", "请选择两张图片")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
