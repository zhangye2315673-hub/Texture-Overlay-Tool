# -*- coding: utf-8 -*-
"""
纹理叠加批处理工具 - Release v1.0
三栏布局：资源库 | 实时画布 | 属性面板
支持实时预览、文件夹扫描、多图追加、预设（含 Base64 内嵌）。
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, Qt, QRect, QSize, QThread, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtGui import QBrush, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# 常量
# =============================================================================
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".psd")
THUMBNAIL_SIZE = 128
PREVIEW_MAX_SIZE = 800
FILL_MODE_STRETCH = "stretch"
FILL_MODE_ASPECT_FILL = "aspect_fill"
FILL_MODE_TILE = "tile"


def _create_thumbnail_icon(image_path: str, size: int = THUMBNAIL_SIZE) -> QPixmap:
    try:
        img = Image.open(image_path)
        img = img.convert("RGBA")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        w, h = img.size
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, w, h, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg)
    except Exception:
        return QPixmap()


class FileListDelegate(QStyledItemDelegate):
    """点击复选框只切换勾选，不改变选中项；点击文件名/图标才切换选中"""

    def editorEvent(self, event, model, option, index):
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            # 复选框区域：左上角 24x24（IconMode 下图标居中，复选框在左上）
            check_rect = QRect(option.rect.x(), option.rect.y(), 24, 24)
            if check_rect.contains(event.pos()):
                state = index.data(Qt.ItemDataRole.CheckStateRole)
                new_state = Qt.CheckState.Unchecked if state == Qt.CheckState.Checked else Qt.CheckState.Checked
                model.setData(index, new_state, Qt.ItemDataRole.CheckStateRole)
                return True
        return super().editorEvent(event, model, option, index)


# =============================================================================
# 混合模式算法（保持不变）
# =============================================================================

def _blend_normal(a, b): return b
def _blend_dissolve(a, b, opacity=1.0):
    h, w = a.shape[:2]
    np.random.seed(42)
    return np.where(np.random.random((h, w))[:, :, np.newaxis] < opacity, b, a)
def _blend_darken(a, b): return np.minimum(a, b)
def _blend_multiply(a, b): return a * b
def _blend_color_burn(a, b):
    eps = 1e-7
    return np.where(b > eps, np.maximum(0, 1 - (1 - a) / b), 0)
def _blend_linear_burn(a, b): return np.maximum(0, a + b - 1)
def _blend_darker_color(a, b):
    lum_a = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    lum_b = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
    return np.where(lum_a[:, :, np.newaxis] <= lum_b[:, :, np.newaxis], a, b)
def _blend_lighten(a, b): return np.maximum(a, b)
def _blend_screen(a, b): return 1 - (1 - a) * (1 - b)
def _blend_color_dodge(a, b): return np.where(b < 1, np.minimum(1, a / (1 - b + 1e-7)), 1)
def _blend_linear_dodge(a, b): return np.minimum(1, a + b)
def _blend_lighter_color(a, b):
    lum_a = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    lum_b = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
    return np.where(lum_a[:, :, np.newaxis] >= lum_b[:, :, np.newaxis], a, b)
def _blend_overlay(a, b): return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
def _blend_soft_light(a, b):
    return np.where(b < 0.5, 2 * a * b + (a * a) * (1 - 2 * b),
                    2 * a * (1 - b) + np.sqrt(np.maximum(a, 0)) * (2 * b - 1))
def _blend_hard_light(a, b): return np.where(b < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
def _blend_vivid_light(a, b):
    return np.where(b >= 0.5, np.clip(1 - (1 - a) / (2 * (b - 0.5) + 1e-7), 0, 1),
                    np.clip(a / (1 - 2 * b + 1e-7), 0, 1))
def _blend_linear_light(a, b):
    return np.clip(np.where(b >= 0.5, a + 2 * (b - 0.5), a + 2 * b - 1), 0, 1)
def _blend_pin_light(a, b): return np.where(b >= 0.5, np.maximum(a, 2 * (b - 0.5)), np.minimum(a, 2 * b))
def _blend_hard_mix(a, b): return np.where(a + b >= 1, 1, 0)
def _blend_difference(a, b): return np.abs(a - b)
def _blend_exclusion(a, b): return np.clip(a + b - 2 * a * b, 0, 1)
def _blend_subtract(a, b): return np.maximum(0, a - b)
def _blend_divide(a, b): return np.where(b > 1e-7, np.minimum(1, a / b), 0)


def _rgb_to_hsl(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cmax, cmin = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2
    s = np.where(delta == 0, 0, np.where(l <= 0.5, delta / (cmax + cmin + 1e-7), delta / (2 - cmax - cmin + 1e-7)))
    h = np.zeros_like(r)
    for m, v in [(cmax == r, ((g - b) / (delta + 1e-7)) % 6), (cmax == g, (b - r) / (delta + 1e-7) + 2), (cmax == b, (r - g) / (delta + 1e-7) + 4)]:
        h = np.where(m & (delta > 0), v, h)
    return h / 6, s, l


def _hsl_to_rgb(h, s, l):
    h = h * 6
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs(h % 2 - 1))
    m = l - c / 2
    masks = [(h >= i) & (h < i + 1) for i in range(6)]
    vals = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
    r = sum(np.where(masks[i], vals[i][0], 0) for i in range(6)) + m
    g = sum(np.where(masks[i], vals[i][1], 0) for i in range(6)) + m
    b = sum(np.where(masks[i], vals[i][2], 0) for i in range(6)) + m
    return np.stack([r, g, b], axis=-1)


def _blend_hue(a, b): return _hsl_to_rgb(*_rgb_to_hsl(b)[:1], *_rgb_to_hsl(a)[1:])
def _blend_saturation(a, b): return _hsl_to_rgb(_rgb_to_hsl(a)[0], _rgb_to_hsl(b)[1], _rgb_to_hsl(a)[2])
def _blend_color(a, b): return _hsl_to_rgb(*_rgb_to_hsl(b)[:2], _rgb_to_hsl(a)[2])
def _blend_luminosity(a, b): return _hsl_to_rgb(*_rgb_to_hsl(a)[:2], _rgb_to_hsl(b)[2])


BLEND_MODES: List[Tuple[str, str, Callable]] = [
    ("组合模式", "正常 (Normal)", _blend_normal),
    ("组合模式", "溶解 (Dissolve)", _blend_dissolve),
    ("变暗组", "变暗 (Darken)", _blend_darken),
    ("变暗组", "正片叠底 (Multiply)", _blend_multiply),
    ("变暗组", "颜色加深 (Color Burn)", _blend_color_burn),
    ("变暗组", "线性加深 (Linear Burn)", _blend_linear_burn),
    ("变暗组", "深色 (Darker Color)", _blend_darker_color),
    ("变亮组", "变亮 (Lighten)", _blend_lighten),
    ("变亮组", "滤色 (Screen)", _blend_screen),
    ("变亮组", "颜色减淡 (Color Dodge)", _blend_color_dodge),
    ("变亮组", "线性减淡 (Linear Dodge)", _blend_linear_dodge),
    ("变亮组", "浅色 (Lighter Color)", _blend_lighter_color),
    ("对比组", "叠加 (Overlay)", _blend_overlay),
    ("对比组", "柔光 (Soft Light)", _blend_soft_light),
    ("对比组", "强光 (Hard Light)", _blend_hard_light),
    ("对比组", "亮光 (Vivid Light)", _blend_vivid_light),
    ("对比组", "线性光 (Linear Light)", _blend_linear_light),
    ("对比组", "点光 (Pin Light)", _blend_pin_light),
    ("对比组", "实色混合 (Hard Mix)", _blend_hard_mix),
    ("差值组", "差值 (Difference)", _blend_difference),
    ("差值组", "排除 (Exclusion)", _blend_exclusion),
    ("差值组", "减去 (Subtract)", _blend_subtract),
    ("差值组", "划分 (Divide)", _blend_divide),
    ("色彩组", "色相 (Hue)", _blend_hue),
    ("色彩组", "饱和度 (Saturation)", _blend_saturation),
    ("色彩组", "颜色 (Color)", _blend_color),
    ("色彩组", "明度 (Luminosity)", _blend_luminosity),
]
BLEND_FUNC_MAP = {name: func for _, name, func in BLEND_MODES}


# =============================================================================
# 纹理配置与处理管线
# =============================================================================
@dataclass
class TextureLayerConfig:
    path: str
    blend_mode: str
    fill_mode: str
    opacity: float


def _prepare_texture(texture: Image.Image, w: int, h: int, fill_mode: str) -> Image.Image:
    tw, th = texture.size
    if fill_mode == FILL_MODE_STRETCH:
        return texture.resize((w, h), Image.Resampling.LANCZOS)
    if fill_mode == FILL_MODE_ASPECT_FILL:
        scale = max(w / tw, h / th)
        nw, nh = int(tw * scale + 0.5), int(th * scale + 0.5)
        scaled = texture.resize((nw, nh), Image.Resampling.LANCZOS)
        return scaled.crop(((nw - w) // 2, (nh - h) // 2, (nw - w) // 2 + w, (nh - h) // 2 + h))
    cols, rows = max(1, (w + tw - 1) // tw), max(1, (h + th - 1) // th)
    tiled = Image.new("RGBA", (cols * tw, rows * th), (0, 0, 0, 0))
    for y in range(rows):
        for x in range(cols):
            tiled.paste(texture, (x * tw, y * th))
    return tiled.crop((0, 0, w, h))


def _apply_single_layer(base_arr: np.ndarray, tex_arr: np.ndarray, blend_mode: str, opacity: float) -> np.ndarray:
    base_rgb, base_alpha = base_arr[:, :, :3], base_arr[:, :, 3]
    tex_rgb = tex_arr[:, :, :3]
    blend_func = BLEND_FUNC_MAP.get(blend_mode, _blend_normal)
    if blend_mode in ("色相 (Hue)", "饱和度 (Saturation)", "颜色 (Color)", "明度 (Luminosity)", "深色 (Darker Color)", "浅色 (Lighter Color)"):
        blended = blend_func(base_rgb, tex_rgb)
    elif blend_mode == "溶解 (Dissolve)":
        blended = blend_func(base_rgb, tex_rgb, opacity)
    else:
        blended = np.stack([blend_func(base_rgb[:, :, c], tex_rgb[:, :, c]) for c in range(3)], axis=-1)
    out_rgb = np.clip(base_rgb * (1 - opacity) + blended * opacity, 0, 1)
    return np.concatenate([out_rgb, base_alpha[:, :, np.newaxis]], axis=-1)


def process_single_image(base_img: Image.Image, texture_configs: List[TextureLayerConfig]) -> np.ndarray:
    """
    可复用的混合管线：对 base_img 依次叠加纹理层，返回 RGBA 数组 (0-1)。
    预览和导出共用此逻辑。
    """
    current = np.array(base_img.convert("RGBA"), dtype=np.float64) / 255.0
    w, h = base_img.size
    original_alpha = current[:, :, 3].copy()
    for cfg in texture_configs:
        if not cfg.path or not os.path.isfile(cfg.path):
            continue
        tex = Image.open(cfg.path).convert("RGBA")
        tex_prep = _prepare_texture(tex, w, h, cfg.fill_mode)
        if tex_prep.size != (w, h):
            tex_prep = tex_prep.resize((w, h), Image.Resampling.LANCZOS)
        tex_arr = np.array(tex_prep, dtype=np.float64) / 255.0
        current = _apply_single_layer(current, tex_arr, cfg.blend_mode, cfg.opacity)
    current[:, :, 3] = original_alpha
    return np.clip(current, 0, 1)


# =============================================================================
# 实时画布（棋盘格 + 居中缩放）
# =============================================================================
def _make_checkerboard_brush(tile=12):
    """创建灰白棋盘格画刷"""
    pix = QPixmap(tile * 2, tile * 2)
    pix.fill(0xFFE0E0E0)
    painter = QPainter(pix)
    painter.fillRect(tile, 0, tile, tile, 0xFFC0C0C0)
    painter.fillRect(0, tile, tile, tile, 0xFFC0C0C0)
    painter.end()
    return QBrush(pix)


class PreviewCanvas(QGraphicsView):
    """灰白棋盘格背景，滚轮缩放，左键 A/B 对比原图，中键拖拽平移"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(_make_checkerboard_brush())
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self._placeholder_text = None
        self._zoom = 1.0
        self._original_pixmap: QPixmap = QPixmap()
        self._processed_pixmap: QPixmap = QPixmap()
        self._pan_start = None

    def set_preview_pixmap(self, pixmap: QPixmap, original_pix: QPixmap = None, reset_view: bool = True):
        self._processed_pixmap = pixmap
        self._original_pixmap = original_pix if original_pix and not original_pix.isNull() else QPixmap()
        if pixmap.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(0, 0, 400, 300)
            if self._placeholder_text is None:
                self._placeholder_text = self._scene.addText("请选择图片预览")
                self._placeholder_text.setDefaultTextColor(Qt.GlobalColor.gray)
                br = self._placeholder_text.boundingRect()
                self._placeholder_text.setPos(200 - br.width() / 2, 150 - br.height() / 2)
        else:
            if self._placeholder_text:
                self._scene.removeItem(self._placeholder_text)
                self._placeholder_text = None
            self._pixmap_item.setPixmap(pixmap)
            self._scene.setSceneRect(self._pixmap_item.boundingRect())
        if reset_view:
            self._zoom = 1.0
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(50, self._zoom))
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and not self._original_pixmap.isNull():
            self._pixmap_item.setPixmap(self._original_pixmap)
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and not self._original_pixmap.isNull():
            self._pixmap_item.setPixmap(self._processed_pixmap)
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self._scene.sceneRect().width() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)


class PreviewCenterWidget(QWidget):
    """画布 + 叠加的加载提示框 + A/B 对比提示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PreviewCanvas()
        layout.addWidget(self.canvas)
        self.loading_label = QLabel("正在渲染 Preview...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet(
            "background: rgba(0,0,0,0.75); color: white; font-size: 14px; "
            "padding: 16px 24px; border-radius: 8px;"
        )
        self.loading_label.setFixedSize(220, 50)
        self.loading_label.setParent(self)
        self.loading_label.hide()
        self.loading_label.raise_()
        self.hint_label = QLabel("按住左键对比原图")
        self.hint_label.setStyleSheet(
            "background: rgba(0,0,0,0.6); color: #ccc; font-size: 11px; "
            "padding: 4px 8px; border-radius: 4px;"
        )
        self.hint_label.setParent(self)
        self.hint_label.adjustSize()
        self.hint_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.loading_label.isVisible():
            w, h = self.loading_label.width(), self.loading_label.height()
            self.loading_label.setGeometry((self.width() - w) // 2, (self.height() - h) // 2, w, h)
        self.hint_label.setGeometry(self.width() - self.hint_label.width() - 12, self.height() - self.hint_label.height() - 12,
                                    self.hint_label.width(), self.hint_label.height())


# =============================================================================
# 纹理图层卡片
# =============================================================================
def _create_layer_thumb(path: str, size=80) -> QPixmap:
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((size * 2, size * 2), Image.Resampling.LANCZOS)
        w, h = img.size
        qimg = QImage(img.tobytes("raw", "RGBA"), w, h, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        if scaled.width() > size or scaled.height() > size:
            x, y = (scaled.width() - size) // 2, (scaled.height() - size) // 2
            return scaled.copy(x, y, size, size)
        return scaled
    except Exception:
        return QPixmap()


class TextureLayerWidget(QFrame):
    delete_requested = Signal(QWidget)
    parameter_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setFixedHeight(120)
        self.setStyleSheet("TextureLayerWidget { background: #f0f0f0; border-radius: 6px; border: 1px solid #ccc; }")
        main = QHBoxLayout(self)
        main.setContentsMargins(8, 6, 8, 6)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(80, 80)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet("background: #ddd; border: 1px dashed #999; border-radius: 4px; color: #666; font-size: 11px;")
        self.thumb_label.setText("无图片")
        main.addWidget(self.thumb_label)

        right = QVBoxLayout()
        r1 = QHBoxLayout()
        self.path_label = QLabel("(未选择)")
        self.path_label.setStyleSheet("font-weight: bold; font-size: 12px; color: gray;")
        r1.addWidget(self.path_label)
        r1.addStretch()
        self.btn_up = QPushButton("↑")
        self.btn_up.setFixedSize(28, 24)
        self.btn_up.setToolTip("上移")
        self.btn_up.clicked.connect(self._emit_parameter_changed)
        r1.addWidget(self.btn_up)
        self.btn_down = QPushButton("↓")
        self.btn_down.setFixedSize(28, 24)
        self.btn_down.setToolTip("下移")
        self.btn_down.clicked.connect(self._emit_parameter_changed)
        r1.addWidget(self.btn_down)
        self.btn_delete = QPushButton("✖")
        self.btn_delete.setFixedSize(28, 24)
        self.btn_delete.setToolTip("删除该图层")
        self.btn_delete.setStyleSheet("color: #c00; font-weight: bold;")
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self))
        self.btn_delete.clicked.connect(self._emit_parameter_changed)
        r1.addWidget(self.btn_delete)
        right.addLayout(r1)

        r2 = QHBoxLayout()
        self.btn_path = QPushButton("更换纹理...")
        self.btn_path.setFixedWidth(90)
        r2.addWidget(self.btn_path)
        self.mode_combo = QComboBox()
        for g, n, _ in BLEND_MODES:
            self.mode_combo.addItem(f"{g} - {n}", n)
        self.mode_combo.setMinimumWidth(120)
        r2.addWidget(self.mode_combo)
        self.fill_combo = QComboBox()
        for t, v in [("拉伸", FILL_MODE_STRETCH), ("等比覆盖", FILL_MODE_ASPECT_FILL), ("平铺", FILL_MODE_TILE)]:
            self.fill_combo.addItem(t, v)
        self.fill_combo.setMinimumWidth(80)
        r2.addWidget(self.fill_combo)
        right.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("不透明度:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        r3.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("80%")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v}%"))
        r3.addWidget(self.opacity_label)
        right.addLayout(r3)

        main.addLayout(right)
        self._path = ""

        self.mode_combo.currentIndexChanged.connect(self._emit_parameter_changed)
        self.fill_combo.currentIndexChanged.connect(self._emit_parameter_changed)
        self.opacity_slider.valueChanged.connect(self._emit_parameter_changed)

    def _emit_parameter_changed(self):
        self.parameter_changed.emit()

    def set_path(self, path: str):
        self._path = path or ""
        self.path_label.setText(Path(path).name if path else "(未选择)")
        self.path_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #333;" if path else "font-weight: bold; font-size: 12px; color: gray;")
        if path:
            p = _create_layer_thumb(path, 80)
            if not p.isNull():
                self.thumb_label.setPixmap(p)
                self.thumb_label.setText("")
            else:
                self.thumb_label.clear()
                self.thumb_label.setText("无图片")
        else:
            self.thumb_label.clear()
            self.thumb_label.setText("无图片")
        self.parameter_changed.emit()

    def get_path(self): return self._path

    def get_config(self) -> TextureLayerConfig:
        return TextureLayerConfig(
            path=self._path,
            blend_mode=self.mode_combo.currentData() or "正常 (Normal)",
            fill_mode=self.fill_combo.currentData() or FILL_MODE_STRETCH,
            opacity=self.opacity_slider.value() / 100.0,
        )

    def apply_config(self, cfg: dict):
        """从预设 dict 恢复参数（path, blend_mode, fill_mode, opacity）"""
        if "path" in cfg and cfg["path"]:
            self.set_path(cfg["path"])
        else:
            self.set_path("")
        if "blend_mode" in cfg:
            idx = self.mode_combo.findData(cfg["blend_mode"])
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
        if "fill_mode" in cfg:
            idx = self.fill_combo.findData(cfg["fill_mode"])
            if idx >= 0:
                self.fill_combo.setCurrentIndex(idx)
        if "opacity" in cfg:
            v = int(round(float(cfg["opacity"]) * 100))
            self.opacity_slider.setValue(max(0, min(100, v)))


# =============================================================================
# 后台处理
# =============================================================================
class ProcessWorker(QThread):
    progress = Signal(int, int)
    log_message = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, image_paths: List[str], layer_configs: List[TextureLayerConfig], output_dir: str):
        super().__init__()
        self.image_paths = image_paths
        self.layer_configs = [c for c in layer_configs if c.path and os.path.isfile(c.path)]
        self.output_dir = output_dir

    def run(self):
        try:
            total, success = len(self.image_paths), 0
            for i, path in enumerate(self.image_paths):
                base = Image.open(path).convert("RGBA")
                result = process_single_image(base, self.layer_configs)
                out_img = Image.fromarray((result * 255).astype(np.uint8), "RGBA")
                out_path = os.path.join(self.output_dir, Path(path).name)
                if os.path.normpath(out_path) == os.path.normpath(path):
                    stem, ext = Path(path).stem, Path(path).suffix
                    out_path = os.path.join(self.output_dir, f"{stem}_textured{ext}")
                out_img.save(out_path, "PNG")
                success += 1
                self.log_message.emit(f"[OK] {Path(path).name}")
                self.progress.emit(i + 1, total)
            self.finished_signal.emit(True, f"完成！成功 {success}/{total} 张")
        except Exception as e:
            self.log_message.emit(f"[错误] {str(e)}")
            self.finished_signal.emit(False, str(e))


# =============================================================================
# 主窗口
# =============================================================================
class TextureOverlayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._file_list: List[str] = []
        self._layer_widgets: List[TextureLayerWidget] = []
        self._preview_base_path: str = ""
        self._preview_cache: Image.Image | None = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("QEA Texture Overlay Tool - Pro v1.0")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 850)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # ----- 左栏：资源库 (20%) -----
        left = QWidget()
        left.setMinimumWidth(200)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        btn_import_row = QHBoxLayout()
        btn_folder = QPushButton("📂 选择文件夹")
        btn_folder.setMinimumHeight(36)
        btn_folder.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; font-weight: bold; border-radius: 5px; } "
            "QPushButton:hover { background: #1976D2; }"
        )
        btn_folder.clicked.connect(self._choose_folder)
        btn_add = QPushButton("📄 添加图片...")
        btn_add.setMinimumHeight(36)
        btn_add.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; font-weight: bold; border-radius: 5px; } "
            "QPushButton:hover { background: #1976D2; }"
        )
        btn_add.clicked.connect(self._add_images)
        btn_import_row.addWidget(btn_folder)
        btn_import_row.addWidget(btn_add)
        left_layout.addLayout(btn_import_row)
        self.file_count_label = QLabel("已加载: 0 个文件")
        self.file_count_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        left_layout.addWidget(self.file_count_label)
        self.file_list = QListWidget()
        self.file_list.setItemDelegate(FileListDelegate(self.file_list))
        self.file_list.setViewMode(QListWidget.IconMode)
        self.file_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.file_list.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.file_list.setSpacing(8)
        self.file_list.setGridSize(QSize(THUMBNAIL_SIZE + 24, THUMBNAIL_SIZE + 36))
        self.file_list.setMovement(QListWidget.Static)
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.setWordWrap(True)
        self.file_list.currentItemChanged.connect(self._on_current_item_changed)
        left_layout.addWidget(self.file_list)
        btn_row = QHBoxLayout()
        for t, f in [("全选", self._select_all), ("全不选", self._deselect_all)]:
            b = QPushButton(t)
            b.setMinimumWidth(55)
            b.clicked.connect(f)
            btn_row.addWidget(b)
        btn_row.addStretch()
        b = QPushButton("刷新")
        b.setMinimumWidth(55)
        b.clicked.connect(lambda: self._refresh_file_list())
        btn_row.addWidget(b)
        b_clear = QPushButton("清空")
        b_clear.setMinimumWidth(55)
        b_clear.setToolTip("清空当前文件列表")
        b_clear.clicked.connect(self._clear_all_files)
        btn_row.addWidget(b_clear)
        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # ----- 中栏：实时画布 (50%) -----
        self.preview_center = PreviewCenterWidget()
        self.preview_center.setMinimumWidth(400)
        self.preview_center.canvas.setStyleSheet("background: #c0c0c0;")
        splitter.addWidget(self.preview_center)

        # ----- 右栏：属性面板 (30%) -----
        right = QWidget()
        right.setMinimumWidth(280)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("当前文件夹（由上方按钮设置）")
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setStyleSheet("background: #f5f5f5; color: #666;")
        right_layout.addWidget(QLabel("当前文件夹:"))
        right_layout.addWidget(self.folder_edit)

        layer_group = QGroupBox("纹理图层（从下到上叠加）")
        layer_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layer_layout = QVBoxLayout(layer_group)
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layer_scroll.setMinimumHeight(200)
        self.layer_container = QWidget()
        self.layer_container_layout = QVBoxLayout(self.layer_container)
        self.layer_container_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_scroll.setWidget(self.layer_container)
        layer_layout.addWidget(self.layer_scroll)
        preset_row = QHBoxLayout()
        btn_save_preset = QPushButton("保存配置")
        btn_save_preset.setMinimumWidth(75)
        btn_save_preset.clicked.connect(self._save_preset)
        btn_load_preset = QPushButton("加载配置")
        btn_load_preset.setMinimumWidth(75)
        btn_load_preset.clicked.connect(self._load_preset)
        preset_row.addWidget(btn_save_preset)
        preset_row.addWidget(btn_load_preset)
        preset_row.addStretch()
        layer_layout.addLayout(preset_row)
        self.btn_add_layer = QPushButton("+ 添加图层")
        self.btn_add_layer.clicked.connect(self._add_layer)
        layer_layout.addWidget(self.btn_add_layer)
        right_layout.addWidget(layer_group, 1)

        op_group = QGroupBox("操作")
        op_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        op_layout = QVBoxLayout(op_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #2196F3; border-radius: 4px; background: #e3f2fd; } "
            "QProgressBar::chunk { background: #2196F3; border-radius: 3px; }"
        )
        op_layout.addWidget(self.progress_bar)
        op_btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.setMinimumHeight(38)
        self.btn_start.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; font-weight: bold; font-size: 13px; "
            "border-radius: 5px; padding: 8px 16px; } "
            "QPushButton:hover { background: #1976D2; } QPushButton:disabled { background: #90CAF9; }"
        )
        self.btn_start.clicked.connect(self._start_process)
        op_btn_row.addWidget(self.btn_start)
        self.btn_open_output = QPushButton("打开输出目录")
        self.btn_open_output.clicked.connect(self._open_output_dir)
        op_btn_row.addWidget(self.btn_open_output)
        op_layout.addLayout(op_btn_row)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(80)
        self.log_edit.setPlaceholderText("处理日志...")
        op_layout.addWidget(self.log_edit)
        right_layout.addWidget(op_group, 0)

        splitter.addWidget(right)
        splitter.setSizes([280, 700, 420])  # 左 20% : 中 50% : 右 30%

        central = QWidget()
        central.setLayout(QVBoxLayout())
        central.layout().addWidget(splitter)
        self.setCentralWidget(central)

        self._add_layer()

    def _add_layer(self):
        w = TextureLayerWidget()
        w.btn_path.clicked.connect(lambda: self._choose_texture_for_layer(w))
        w.delete_requested.connect(self._remove_layer)
        w.parameter_changed.connect(self._update_preview)
        w.btn_up.clicked.connect(lambda: self._move_layer(w, -1))
        w.btn_down.clicked.connect(lambda: self._move_layer(w, 1))
        self._layer_widgets.append(w)
        self.layer_container_layout.addWidget(w)
        self._update_preview()

    def _remove_layer(self, widget: QWidget):
        if widget in self._layer_widgets and len(self._layer_widgets) > 1:
            self._layer_widgets.remove(widget)
            self.layer_container_layout.removeWidget(widget)
            widget.deleteLater()
            self._update_preview()

    def _choose_texture_for_layer(self, w: TextureLayerWidget):
        path, _ = QFileDialog.getOpenFileName(self, "选择纹理", filter="图片 (*.png *.jpg *.jpeg *.bmp *.tga *.tif *.tiff)")
        if path:
            w.set_path(path)

    def _move_layer(self, w: TextureLayerWidget, delta: int):
        idx = self._layer_widgets.index(w)
        new_idx = idx + delta
        if 0 <= new_idx < len(self._layer_widgets):
            self._layer_widgets.remove(w)
            self._layer_widgets.insert(new_idx, w)
            self.layer_container_layout.removeWidget(w)
            self.layer_container_layout.insertWidget(new_idx, w)
            self._update_preview()

    def _save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存配置", filter="JSON (*.json)")
        if not path:
            return
        configs = [w.get_config() for w in self._layer_widgets]
        data = []
        for c in configs:
            item = {"path": c.path, "blend_mode": c.blend_mode, "fill_mode": c.fill_mode, "opacity": c.opacity}
            if c.path and os.path.isfile(c.path):
                try:
                    with open(c.path, "rb") as f:
                        item["base64_data"] = base64.b64encode(f.read()).decode("ascii")
                except Exception:
                    pass
            data.append(item)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "完成", f"配置已保存到 {path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载配置", filter="JSON (*.json)")
        if path:
            self._load_preset_from_path(path)

    def _load_preset_from_path(self, path: str):
        """从指定路径加载预设（含 Base64 内嵌图片）"""
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载失败: {e}")
            return
        while len(self._layer_widgets) > 1:
            w = self._layer_widgets[-1]
            self._layer_widgets.remove(w)
            self.layer_container_layout.removeWidget(w)
            w.deleteLater()
        for i, cfg in enumerate(data):
            if i >= len(self._layer_widgets):
                self._add_layer()
            layer_cfg = dict(cfg)
            path_val = layer_cfg.get("path", "")
            if path_val and not os.path.isfile(path_val) and "base64_data" in layer_cfg:
                try:
                    raw = base64.b64decode(layer_cfg["base64_data"])
                    fd, tmp = tempfile.mkstemp(suffix=Path(path_val).suffix or ".png")
                    os.close(fd)
                    with open(tmp, "wb") as f:
                        f.write(raw)
                    layer_cfg["path"] = tmp
                except Exception:
                    layer_cfg["path"] = ""
            self._layer_widgets[i].apply_config(layer_cfg)
        while len(self._layer_widgets) > max(1, len(data)):
            w = self._layer_widgets[-1]
            self._layer_widgets.remove(w)
            self.layer_container_layout.removeWidget(w)
            w.deleteLater()
        if not data and self._layer_widgets:
            self._layer_widgets[0].apply_config({})
        self._update_preview(reset_view=True)

    def _choose_folder(self):
        """📂 选择文件夹：扫描文件夹内所有图片"""
        path = QFileDialog.getExistingDirectory(self, "选择待处理图片文件夹")
        if path:
            self.folder_edit.setText(path)
            self._refresh_file_list(path)

    def _add_images(self):
        """📄 添加图片：多选追加到列表（不清空已有）"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加图片", filter="图片 (*.png *.jpg *.jpeg *.bmp *.tga *.tif *.tiff *.psd)"
        )
        if not paths:
            return
        existing = set(self._file_list)
        added = [p for p in paths if p not in existing]
        if not added:
            return
        if not self.folder_edit.text().strip():
            self.folder_edit.setText(str(Path(added[0]).parent))
        for p in added:
            self._file_list.append(p)
            icon = _create_thumbnail_icon(p)
            item = QListWidgetItem(QIcon(icon), Path(p).name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.file_list.addItem(item)
        self.file_count_label.setText(f"已加载: {len(self._file_list)} 个文件")
        if not self._preview_base_path or self._preview_base_path not in self._file_list:
            self.file_list.setCurrentRow(len(self._file_list) - len(added))
            self._preview_base_path = added[0]
            self._preview_cache = None
        self._update_preview(reset_view=not self._preview_base_path)

    def _refresh_file_list(self, folder_path: str = None):
        path = folder_path if folder_path is not None else self.folder_edit.text().strip()
        self._file_list.clear()
        self.file_list.clear()
        self._preview_base_path = ""
        self._preview_cache = None
        if not path or not os.path.isdir(path):
            self.file_count_label.setText("已加载: 0 个文件")
            self._update_preview(reset_view=True)
            return
        for f in sorted(os.listdir(path)):
            if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                full = os.path.join(path, f)
                self._file_list.append(full)
                icon = _create_thumbnail_icon(full)
                item = QListWidgetItem(QIcon(icon), f)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(Qt.ItemDataRole.UserRole, full)
                self.file_list.addItem(item)
        self.file_count_label.setText(f"已加载: {len(self._file_list)} 个文件")
        if self._file_list:
            self.file_list.setCurrentRow(0)
            self._preview_base_path = self._file_list[0]
            self._preview_cache = None
        else:
            self._preview_base_path = ""
            self._preview_cache = None
        self._update_preview(reset_view=True)

    def _on_current_item_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        """仅当选中项改变时切换预览，勾选/取消勾选不触发"""
        if current is None:
            self._preview_base_path = ""
            self._preview_cache = None
            self._update_preview()
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        if path and path != self._preview_base_path:
            self._preview_base_path = path
            self._preview_cache = None
            self._update_preview(reset_view=True)

    def _update_preview(self, reset_view: bool = False):
        """实时预览：仅处理当前选中图片，参数变化时只刷新像素不重置视图"""
        canvas = self.preview_center.canvas
        loading = self.preview_center.loading_label

        if not self._preview_base_path or not os.path.isfile(self._preview_base_path):
            canvas.set_preview_pixmap(QPixmap())
            return

        loading.setGeometry(
            (self.preview_center.width() - loading.width()) // 2,
            (self.preview_center.height() - loading.height()) // 2,
            loading.width(), loading.height()
        )
        loading.show()
        loading.raise_()
        QApplication.processEvents()

        try:
            configs = [w.get_config() for w in self._layer_widgets]
            valid = [c for c in configs if c.path and os.path.isfile(c.path)]
            if self._preview_cache is None or self._preview_base_path != getattr(self, "_cache_path", ""):
                try:
                    base = Image.open(self._preview_base_path).convert("RGBA")
                    base.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE), Image.Resampling.LANCZOS)
                    self._preview_cache = base.copy()
                    self._cache_path = self._preview_base_path
                except Exception:
                    canvas.set_preview_pixmap(QPixmap())
                    return
            base = self._preview_cache.copy()
            result = process_single_image(base, valid)
            arr = np.ascontiguousarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg)
            base_arr = np.ascontiguousarray(np.array(base, dtype=np.uint8))
            bw, bh = base.size[0], base.size[1]
            base_qimg = QImage(base_arr.data, bw, bh, bw * 4, QImage.Format.Format_RGBA8888)
            orig_pix = QPixmap.fromImage(base_qimg.copy())
            canvas.set_preview_pixmap(pix, original_pix=orig_pix, reset_view=reset_view)
        finally:
            loading.hide()

    def _clear_all_files(self):
        """一键清空当前文件列表"""
        self._file_list.clear()
        self.file_list.clear()
        self._preview_base_path = ""
        self._preview_cache = None
        self.file_count_label.setText("已加载: 0 个文件")
        self._update_preview(reset_view=True)

    def _select_all(self):
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self):
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _get_checked_paths(self) -> List[str]:
        return [self.file_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.file_list.count())
                if self.file_list.item(i).checkState() == Qt.CheckState.Checked
                and self.file_list.item(i).data(Qt.ItemDataRole.UserRole)]

    def _get_layer_configs(self) -> List[TextureLayerConfig]:
        return [w.get_config() for w in self._layer_widgets]

    def _get_output_dir(self) -> str:
        folder = self.folder_edit.text().strip()
        paths = self._get_checked_paths()
        if folder and os.path.isdir(folder):
            return os.path.join(folder, "_Output")
        if paths:
            return os.path.join(str(Path(paths[0]).parent), "_Output")
        return ""

    def _open_output_dir(self):
        out = self._get_output_dir()
        if not out:
            QMessageBox.warning(self, "提示", "请先选择源文件夹或勾选图片。")
            return
        if not os.path.isdir(out):
            QMessageBox.warning(self, "提示", f"输出目录不存在：{out}\n请先执行一次处理。")
            return
        if sys.platform == "win32":
            os.startfile(out)
        elif sys.platform == "darwin":
            subprocess.run(["open", out])
        else:
            subprocess.run(["xdg-open", out])

    def _start_process(self):
        paths = self._get_checked_paths()
        configs = self._get_layer_configs()
        valid = [c for c in configs if c.path and os.path.isfile(c.path)]
        out = self._get_output_dir()
        if not paths:
            QMessageBox.warning(self, "提示", "未勾选任何图片。")
            return
        if not valid:
            QMessageBox.warning(self, "提示", "请至少添加一个有效的纹理图层。")
            return
        os.makedirs(out, exist_ok=True)
        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self.log_edit.appendPlainText(f"开始处理 {len(paths)} 张...")
        self.worker = ProcessWorker(paths, valid, out)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._on_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, cur: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(100 * cur / total))
            self.progress_bar.setFormat(f"{cur}/{total}")

    def _on_log(self, msg: str):
        self.log_edit.appendPlainText(msg)

    def _on_finished(self, success: bool, msg: str):
        self.btn_start.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("完成")
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "错误", msg)


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    win = TextureOverlayWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
